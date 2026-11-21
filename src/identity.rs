use rustc_hash::FxHashMap;
use std::sync::Mutex;

const EXTERNAL: u64 = 1 << 63;
const POOLS: usize = 64;
#[derive(Default)]
struct Pool {
    ids: FxHashMap<Vec<u8>, u64>,
    names: Vec<Vec<u8>>,
}

pub struct Symbols {
    prefix: u8,
    pools: [Mutex<Pool>; POOLS],
}
impl Symbols {
    pub fn new(prefix: u8) -> Self {
        Self {
            prefix,
            pools: std::array::from_fn(|_| Mutex::default()),
        }
    }
    #[inline]
    pub fn encode(&self, s: &[u8]) -> u64 {
        if (1..=9).contains(&s.len()) && s[0] == self.prefix && s.last() != Some(&0) {
            let mut b = [0; 8];
            if s.len() == 9 {
                b.copy_from_slice(&s[1..])
            } else {
                b[..s.len() - 1].copy_from_slice(&s[1..]);
            }
            let n = u64::from_le_bytes(b);
            if n < EXTERNAL {
                return n;
            }
        }
        use std::hash::{BuildHasher, Hash, Hasher};
        let mut h = rustc_hash::FxBuildHasher.build_hasher();
        s.hash(&mut h);
        let slot = (h.finish() as usize) % POOLS;
        let mut p = self.pools[slot].lock().unwrap();
        if let Some(&id) = p.ids.get(s) {
            return id;
        }
        let id = EXTERNAL + (p.names.len() * POOLS + slot) as u64;
        p.names.push(s.to_vec());
        p.ids.insert(s.to_vec(), id);
        id
    }
    pub fn decode(&self, id: u64) -> Vec<u8> {
        if id < EXTERNAL {
            let mut s = Vec::with_capacity(9);
            s.push(self.prefix);
            s.extend_from_slice(
                &id.to_le_bytes()[..(64 - id.leading_zeros() as usize).div_ceil(8)],
            );
            s
        } else {
            let token = (id - EXTERNAL) as usize;
            self.pools[token % POOLS].lock().unwrap().names[token / POOLS].clone()
        }
    }
    pub fn least(&self, a: u64, b: u64) -> u64 {
        if a < EXTERNAL && b < EXTERNAL {
            if a.swap_bytes() < b.swap_bytes() {
                a
            } else {
                b
            }
        } else if self.decode(a) < self.decode(b) {
            a
        } else {
            b
        }
    }
}

// The common ID path never hashes or compares input text after parsing.
#[inline]
pub fn mix(mut x: u64) -> usize {
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8feb86659fd93);
    (x ^ (x >> 32)) as usize
}
pub const EMPTY: u64 = u64::MAX;
pub struct IdMap<const SKIP: usize> {
    slots: Vec<(u64, u64)>,
    len: usize,
}
impl<const SKIP: usize> IdMap<SKIP> {
    pub fn new(estimate: usize) -> Self {
        Self {
            slots: vec![(EMPTY, 0); (estimate + estimate / 2).max(16).next_power_of_two()],
            len: 0,
        }
    }
    #[inline]
    fn locate(&self, key: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut at = (mix(key) >> SKIP) & mask;
        loop {
            let k = self.slots[at].0;
            if k == key || k == EMPTY {
                return at;
            }
            at = (at + 1) & mask;
        }
    }
    pub fn find(&self, key: u64) -> Option<u64> {
        let (k, v) = self.slots[self.locate(key)];
        (k != EMPTY).then_some(v)
    }
    pub fn insert_id(&mut self, key: u64) -> (usize, bool) {
        let mut pos = self.locate(key);
        if self.slots[pos].0 != EMPTY {
            return (self.slots[pos].1 as usize, false);
        }
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            let n = self.slots.len() * 2;
            let old = std::mem::replace(&mut self.slots, vec![(EMPTY, 0); n]);
            for (k, v) in old {
                if k != EMPTY {
                    let p = self.locate(k);
                    self.slots[p] = (k, v);
                }
            }
            pos = self.locate(key);
        }
        let i = self.len;
        self.len += 1;
        self.slots[pos] = (key, i as u64);
        (i, true)
    }
    pub fn replace_values(&mut self, mut f: impl FnMut(usize) -> u64) {
        for (k, v) in &mut self.slots {
            if *k != EMPTY {
                *v = f(*v as usize);
            }
        }
    }
}

pub struct Links {
    ids: IdMap<0>,
    parents: Vec<i32>,
    names: Vec<u64>,
}
impl Links {
    pub fn new(n: usize) -> Self {
        Self {
            ids: IdMap::new(n),
            parents: Vec::with_capacity(n),
            names: Vec::with_capacity(n),
        }
    }
    fn index(&mut self, id: u64) -> usize {
        let (i, new) = self.ids.insert_id(id);
        if new {
            self.parents.push(-1);
            self.names.push(id);
        }
        i
    }
    fn root(parents: &mut [i32], mut i: usize) -> usize {
        while parents[i] >= 0 {
            let j = parents[i] as usize;
            if parents[j] >= 0 {
                parents[i] = parents[j];
            }
            i = j;
        }
        i
    }
    pub fn join(&mut self, a: &[u8], b: &[u8], symbols: &Symbols) {
        if a.is_empty() || b.is_empty() {
            return;
        }
        let ia = self.index(symbols.encode(a));
        let ib = self.index(symbols.encode(b));
        let mut i = Self::root(&mut self.parents, ia);
        let mut j = Self::root(&mut self.parents, ib);
        if i == j {
            return;
        }
        if self.parents[i] > self.parents[j] {
            std::mem::swap(&mut i, &mut j);
        }
        self.parents[i] += self.parents[j];
        self.parents[j] = i as i32;
        self.names[i] = symbols.least(self.names[i], self.names[j]);
    }
    pub fn finish(mut self) -> IdMap<0> {
        self.ids
            .replace_values(|i| self.names[Self::root(&mut self.parents, i)]);
        self.ids
    }
}
