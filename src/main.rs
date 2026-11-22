mod identity;
mod input;
mod output;

use identity::{EMPTY, IdMap, Links, Symbols, mix};
use input::{DAY, Dates, WEEK};
use parquet::data_type::ByteArray;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

const BINS: usize = 64;
type Bins<T> = [Vec<T>; BINS];
fn buckets<T>(cap: usize) -> Bins<T> {
    std::array::from_fn(|_| Vec::with_capacity(cap))
}
fn transpose<T>(chunks: Vec<Bins<T>>) -> Bins<Vec<T>> {
    let mut result = buckets(chunks.len());
    for parts in chunks {
        for (out, part) in result.iter_mut().zip(parts) {
            if !part.is_empty() {
                out.push(part);
            }
        }
    }
    result
}
trait EventValues: Copy + Default + Send + Sync {
    fn new(base: i64, t: i64, price: u32, category: u32) -> Self;
    fn time(self, base: i64) -> i64;
    fn price(self) -> u32;
    fn category(self) -> u32;
}

// The compact format is selected once from the actual window and category
// dictionary. A validated purchase price needs at most 24 bits.
#[derive(Clone, Copy, Default)]
struct CompactEvent {
    millis: u32,
    purchase: u32,
}
impl EventValues for CompactEvent {
    fn new(base: i64, t: i64, price: u32, category: u32) -> Self {
        Self {
            millis: (t - base) as u32,
            purchase: (category << 24) | price,
        }
    }
    fn time(self, base: i64) -> i64 {
        base + i64::from(self.millis)
    }
    fn price(self) -> u32 {
        self.purchase & 0x00ff_ffff
    }
    fn category(self) -> u32 {
        self.purchase >> 24
    }
}
#[derive(Clone, Copy, Default)]
struct WideEvent {
    t: i64,
    price: u32,
    category: u32,
}
impl EventValues for WideEvent {
    fn new(_: i64, t: i64, price: u32, category: u32) -> Self {
        Self { t, price, category }
    }
    fn time(self, _: i64) -> i64 {
        self.t
    }
    fn price(self) -> u32 {
        self.price
    }
    fn category(self) -> u32 {
        self.category
    }
}
#[derive(Clone, Copy)]
struct Event<D> {
    user: u64,
    data: D,
}
#[derive(Clone, Copy)]
struct Record<D> {
    id: u64,
    ingest: i64,
    event: Event<D>,
}
#[derive(Clone, Copy, Default)]
struct Ad<'a> {
    t: i64,
    id: &'a [u8],
    campaign: u32,
}
#[derive(Clone, Copy)]
struct Click<'a> {
    user: u64,
    data: Ad<'a>,
}
#[derive(Default, Clone, Copy)]
pub struct Counts {
    sessions: i64,
    purchasing: i64,
    revenue: i64,
    assists: i64,
}
#[derive(Default)]
pub struct DayRow {
    user: u32,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u32>,
}
pub struct Campaign {
    id: ByteArray,
    channel: ByteArray,
}

fn estimate(chunk: &[u8]) -> usize {
    let n = chunk.len().min(4096);
    if n == 0 {
        return 0;
    }
    let rows = memchr::memchr_iter(b'\n', &chunk[..n]).count();
    (rows * chunk.len().min(512 * 1024) / n / BINS) * 5 / 4 + 2
}

fn dedup<D: EventValues>(chunks: Vec<Vec<Record<D>>>, links: &IdMap<0>) -> Bins<Event<D>> {
    let n: usize = chunks.iter().map(Vec::len).sum();
    let fingerprint_bits = (n.max(16) * 4).next_power_of_two();
    let mut flags = vec![(0u64, 0u64); fingerprint_bits / 64];
    let index = |id| (mix(id) / BINS) & (fingerprint_bits - 1);
    let mut revisits = 0usize;
    for chunk in &chunks {
        for record in chunk {
            let i = index(record.id);
            let bit = 1u64 << (i % 64);
            let (seen, multiple) = &mut flags[i / 64];
            let repeated = *seen & bit;
            *multiple |= repeated;
            *seen |= bit;
            revisits += usize::from(repeated != 0);
        }
    }
    // Every slot seen exactly once has one globally unique ID. All records
    // landing in a repeated slot still compare full IDs and ingestion times.
    // k rows in a repeated slot cause k-1 revisits, so k <= 2*(k-1).
    let capacity = (revisits * 3 + 1).next_power_of_two();
    let mut table = vec![u32::MAX; capacity];
    let mut delayed = Vec::<Record<D>>::with_capacity(2 * revisits);
    let mut output = buckets((n / BINS) * 9 / 8);
    let mut emit = |e: Event<D>| {
        if e.user != EMPTY {
            let user = links.find(e.user).unwrap_or(e.user);
            output[mix(user) % BINS].push(Event { user, data: e.data });
        }
    };
    for chunk in chunks {
        for record in chunk {
            let fp = index(record.id);
            if flags[fp / 64].1 & (1u64 << (fp % 64)) == 0 {
                emit(record.event);
                continue;
            }
            let mut pos = (mix(record.id) / BINS) & (capacity - 1);
            loop {
                let loc = table[pos];
                if loc == u32::MAX {
                    table[pos] = u32::try_from(delayed.len()).unwrap();
                    delayed.push(record);
                    break;
                }
                let old = &mut delayed[loc as usize];
                if old.id == record.id {
                    if record.ingest < old.ingest {
                        *old = record;
                    }
                    break;
                }
                pos = (pos + 1) & (capacity - 1);
            }
        }
    }
    for record in delayed {
        emit(record.event);
    }
    output
}

struct Timeline<T> {
    offsets: Vec<usize>,
    values: Vec<T>,
}
fn timeline<T, U: Copy + Default>(
    chunks: Vec<Vec<T>>,
    known: usize,
    mut identify: impl FnMut(&T) -> Option<u32>,
    project: impl Fn(&T) -> U,
) -> Timeline<U> {
    let n: usize = chunks.iter().map(Vec::len).sum();
    let mut owners = Vec::with_capacity(n);
    let mut counts = vec![0usize; known];
    for ch in &chunks {
        for row in ch {
            let id = identify(row).unwrap_or(u32::MAX);
            owners.push(id);
            if id == u32::MAX {
                continue;
            }
            if id as usize == counts.len() {
                counts.push(0);
            }
            counts[id as usize] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut next = 0;
    for count in &mut counts {
        offsets.push(next);
        let len = *count;
        *count = next;
        next += len;
    }
    offsets.push(next);
    let mut values = vec![U::default(); next];
    for (row, id) in chunks.into_iter().flatten().zip(owners) {
        if id != u32::MAX {
            let at = &mut counts[id as usize];
            values[*at] = project(&row);
            *at += 1;
        }
    }
    Timeline { offsets, values }
}

type Days = FxHashMap<i32, Vec<Counts>>;
fn aggregate<D: EventValues>(
    base: i64,
    bin: usize,
    event_chunks: Vec<Vec<Event<D>>>,
    click_chunks: Vec<Vec<Click<'_>>>,
    users: &Symbols,
    categories: &[ByteArray],
    campaigns: usize,
    output: &Path,
) -> Days {
    let mut people = IdMap::<6>::new(event_chunks.iter().map(Vec::len).sum::<usize>() / 24);
    let mut names = Vec::new();
    let mut events = timeline(
        event_chunks,
        0,
        |e| {
            let (id, new) = people.insert_id(e.user);
            if new {
                names.push(ByteArray::from(users.decode(e.user)));
            }
            Some(id as u32)
        },
        |e| e.data,
    );
    let mut clicks = timeline(
        click_chunks,
        names.len(),
        |a| people.find(a.user).map(|i| i as u32),
        |a| a.data,
    );
    let mut daily = Days::default();
    let mut output_rows = Vec::with_capacity(names.len() * 3);
    let mut active = Vec::<u32>::new();
    let mut position = vec![0; campaigns];
    let mut counts = vec![0u32; campaigns];
    let mut category_revenue = vec![0i64; categories.len()];
    let mut touched = Vec::<u32>::new();
    for person in 0..names.len() {
        for c in active.drain(..) {
            counts[c as usize] = 0;
        }
        let ev = &mut events.values[events.offsets[person]..events.offsets[person + 1]];
        let ad = &mut clicks.values[clicks.offsets[person]..clicks.offsets[person + 1]];
        ev.sort_unstable_by_key(|e| e.time(base));
        ad.sort_unstable_by(|a, b| a.t.cmp(&b.t).then_with(|| b.id.cmp(a.id)));
        let (mut e, mut left, mut right) = (0, 0, 0);
        let mut last_day = None;
        let mut best = 0;
        while e < ev.len() {
            let start = ev[e].time(base);
            let day = start.div_euclid(DAY) as i32;
            if last_day != Some(day) {
                output_rows.push(DayRow {
                    user: person as u32,
                    day,
                    ..Default::default()
                });
                last_day = Some(day);
                for cat in touched.drain(..) {
                    category_revenue[cat as usize] = 0;
                }
                best = 0;
            }
            let user = output_rows.last_mut().unwrap();
            user.sessions += 1;
            let first = e;
            let mut revenue = 0;
            loop {
                let row = ev[e];
                if row.price() != 0 {
                    let sum = &mut category_revenue[row.category() as usize];
                    if *sum == 0 {
                        touched.push(row.category());
                    }
                    *sum += row.price() as i64;
                    revenue += row.price() as i64;
                    if *sum > best
                        || (*sum == best && user.top.is_none_or(|cat| row.category() < cat))
                    {
                        best = *sum;
                        user.top = Some(row.category());
                    }
                }
                e += 1;
                if e == ev.len() || ev[e].time(base) - row.time(base) > 1_800_000 {
                    break;
                }
            }
            user.events += (e - first) as i64;
            user.revenue += revenue;
            while right < ad.len() && ad[right].t <= start {
                let c = ad[right].campaign as usize;
                if counts[c] == 0 {
                    position[c] = active.len();
                    active.push(c as u32);
                }
                counts[c] += 1;
                right += 1;
            }
            while left < right && ad[left].t < start - WEEK {
                let c = ad[left].campaign as usize;
                counts[c] -= 1;
                if counts[c] == 0 {
                    let slot = position[c];
                    active.swap_remove(slot);
                    if slot < active.len() {
                        position[active[slot] as usize] = slot;
                    }
                }
                left += 1;
            }
            let credit = if left < right {
                ad[right - 1].campaign as usize
            } else {
                0
            };
            let totals = daily
                .entry(day)
                .or_insert_with(|| vec![Counts::default(); campaigns]);
            totals[credit].sessions += 1;
            totals[credit].purchasing += i64::from(revenue > 0);
            totals[credit].revenue += revenue;
            for &c in &active {
                totals[c as usize].assists += 1;
            }
        }
    }
    if bin == 0 || !output_rows.is_empty() {
        output::users(
            &output.join("user_daily").join(format!("{bin}.parquet")),
            &names,
            &output_rows,
            categories,
        );
    }
    daily
}

fn run(data: &Path, output: &Path) {
    let profile = env::var_os("SHOP_PROFILE").is_some();
    let mut last = Instant::now();
    let mark = |name: &str| {
        if profile {
            eprintln!("{name} {:.4}s", last.elapsed().as_secs_f64());
            last = Instant::now();
        }
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(data.join("manifest.json")).unwrap()).unwrap();
    let mut dates = Dates::new();
    let start = dates
        .time(manifest["window_start"].as_str().unwrap().as_bytes())
        .unwrap();
    let end = dates
        .time(manifest["window_end"].as_str().unwrap().as_bytes())
        .unwrap();
    let prods = fs::read(data.join("products.csv")).unwrap();
    let mut pairs = Vec::new();
    input::each::<2>(input::after_header(&prods), |f| pairs.push(f));
    let mut category_names = vec![b"unknown".as_slice()];
    category_names.extend(pairs.iter().map(|f| f[1]));
    category_names.sort_unstable();
    category_names.dedup();
    let unknown = category_names
        .binary_search(&b"unknown".as_slice())
        .unwrap() as u32;
    let product_map: FxHashMap<_, _> = pairs
        .into_iter()
        .map(|[k, v]| (k, category_names.binary_search(&v).unwrap() as u32))
        .collect();
    let categories: Vec<_> = category_names
        .iter()
        .map(|s| ByteArray::from(s.to_vec()))
        .collect();
    let camp = fs::read(data.join("campaigns.csv")).unwrap();
    let mut campaigns = vec![Campaign {
        id: ByteArray::from("organic"),
        channel: ByteArray::from("organic"),
    }];
    let mut campaign_ids = FxHashMap::default();
    input::each::<2>(input::after_header(&camp), |[id, channel]| {
        campaign_ids.insert(id, campaigns.len() as u32);
        campaigns.push(Campaign {
            id: ByteArray::from(id.to_vec()),
            channel: ByteArray::from(channel.to_vec()),
        });
    });
    let dims = Dimensions {
        start,
        end,
        unknown,
        product_map,
        campaign_ids,
        categories,
        campaigns,
    };
    if end.abs_diff(start) <= u64::from(u32::MAX) && dims.categories.len() <= 256 {
        process::<CompactEvent>(data, output, &dims, mark);
    } else {
        process::<WideEvent>(data, output, &dims, mark);
    }
}

struct Dimensions<'a> {
    start: i64,
    end: i64,
    unknown: u32,
    product_map: FxHashMap<&'a [u8], u32>,
    campaign_ids: FxHashMap<&'a [u8], u32>,
    categories: Vec<ByteArray>,
    campaigns: Vec<Campaign>,
}

fn process<D: EventValues>(
    data: &Path,
    output: &Path,
    dims: &Dimensions,
    mut mark: impl FnMut(&str),
) {
    let (start, end) = (dims.start, dims.end);
    let events = input::Files::open(&data.join("events")).unwrap();
    let links = input::Files::open(&data.join("links")).unwrap();
    let clicks = input::Files::open(&data.join("clicks")).unwrap();
    let users = Symbols::new(b'u');
    let ids = Symbols::new(b'e');
    mark("map inputs");
    let (identity, event_parts) = rayon::join(
        || {
            let mut tree = Links::new(links.bytes() / 20);
            for chunk in links.chunks() {
                input::each::<2>(chunk.rows(), |[a, b]| tree.join(a, b, &users));
            }
            tree.finish()
        },
        || {
            events
                .chunks()
                .par_iter()
                .map(|chunk| {
                    let bytes = chunk.rows();
                    let mut dates = Dates::new();
                    let mut parts = buckets(estimate(bytes));
                    input::each::<7>(bytes, |[eid, uid, ts, ingest, kind, product, price]| {
                        if eid.is_empty() {
                            return;
                        }
                        let Some(ingest) = dates.ingest(ingest) else {
                            return;
                        };
                        let id = ids.encode(eid);
                        let data = (|| {
                            if uid.is_empty() || product.is_empty() {
                                return None;
                            }
                            let t = dates.time(ts)?;
                            if t < start || t >= end || ingest > t + DAY {
                                return None;
                            }
                            let (price, category) = match kind {
                                b"purchase" => (
                                    input::price(price)?,
                                    *dims.product_map.get(product).unwrap_or(&dims.unknown),
                                ),
                                b"cart" | b"view" => (0, 0),
                                _ => return None,
                            };
                            Some(Event {
                                user: users.encode(uid),
                                data: D::new(start, t, price, category),
                            })
                        })();
                        parts[mix(id) % BINS].push(Record {
                            id,
                            ingest,
                            event: data.unwrap_or(Event {
                                user: EMPTY,
                                data: D::default(),
                            }),
                        });
                    });
                    parts
                })
                .collect::<Vec<_>>()
        },
    );
    mark("resolve links and parse events");
    let (ev, ads) = rayon::join(
        || {
            let raw = transpose(event_parts);
            transpose(
                raw.into_par_iter()
                    .map(|chunks| dedup(chunks, &identity))
                    .collect(),
            )
        },
        || {
            let click_slices = clicks.chunks();
            let parts = click_slices
                .into_par_iter()
                .map(|chunk| {
                    let bytes = chunk.rows();
                    let mut dates = Dates::new();
                    let mut parts = buckets(estimate(bytes));
                    input::each::<4>(bytes, |[id, user, campaign, ts]| {
                        if id.is_empty() || user.is_empty() {
                            return;
                        }
                        let Some(&campaign) = dims.campaign_ids.get(campaign) else {
                            return;
                        };
                        let Some(t) = dates.time(ts) else {
                            return;
                        };
                        if t < start - WEEK || t >= end {
                            return;
                        }
                        let raw = users.encode(user);
                        let user = identity.find(raw).unwrap_or(raw);
                        parts[mix(user) % BINS].push(Click {
                            user,
                            data: Ad { t, id, campaign },
                        });
                    });
                    parts
                })
                .collect();
            transpose(parts)
        },
    );
    mark("deduplicate, partition and prepare clicks");
    fs::create_dir_all(output.join("user_daily")).unwrap();
    fs::create_dir_all(output.join("campaign_daily")).unwrap();
    let days: Vec<_> = ev
        .into_par_iter()
        .zip(ads)
        .enumerate()
        .map(|(i, (events, clicks))| {
            aggregate(
                start,
                i,
                events,
                clicks,
                &users,
                &dims.categories,
                dims.campaigns.len(),
                output,
            )
        })
        .collect();
    let mut all = Days::default();
    for shard in days {
        for (day, rows) in shard {
            let totals = all
                .entry(day)
                .or_insert_with(|| vec![Counts::default(); dims.campaigns.len()]);
            for (a, r) in totals.iter_mut().zip(rows) {
                a.sessions += r.sessions;
                a.purchasing += r.purchasing;
                a.revenue += r.revenue;
                a.assists += r.assists;
            }
        }
    }
    let rows = all
        .into_iter()
        .flat_map(|(day, rows)| {
            rows.into_iter()
                .enumerate()
                .filter(|(_, r)| r.sessions != 0 || r.assists != 0)
                .map(move |(id, r)| (day, id, r))
        })
        .collect::<Vec<_>>();
    output::campaigns(
        &output.join("campaign_daily/0.parquet"),
        &rows,
        &dims.campaigns,
    );
    mark("group timelines, aggregate and write parquet");
}
fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    let dir = |i: usize, key: &str, default: &str| {
        PathBuf::from(
            args.get(i)
                .cloned()
                .or_else(|| env::var_os(key))
                .unwrap_or_else(|| default.into()),
        )
    };
    let n = env::var("NCPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 64);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap();
    run(
        &dir(1, "DATA_DIR", "/data"),
        &dir(2, "OUTPUT_DIR", "/output"),
    );
}
