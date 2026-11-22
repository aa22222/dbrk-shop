use memchr::{memchr, memchr_iter};
use memmap2::Mmap;
use std::{fs, io, path::Path};

pub const CHUNK_BYTES: usize = 2 * 1024 * 1024;

pub struct Files(Vec<Mmap>);

impl Files {
    pub fn open(dir: &Path) -> io::Result<Self> {
        let mut maps = Vec::new();
        for entry in fs::read_dir(dir)? {
            let p = entry?.path();
            if p.extension().is_some_and(|extension| extension == "csv") {
                let f = fs::File::open(p)?;
                if f.metadata()?.len() != 0 {
                    // Scorer input remains mounted read-only for the process lifetime.
                    maps.push(unsafe { Mmap::map(&f)? });
                }
            }
        }
        Ok(Self(maps))
    }

    pub fn bytes(&self) -> usize {
        self.0.iter().map(|m| m.len()).sum()
    }

    pub fn chunks(&self) -> Vec<Chunk<'_>> {
        let mut out = Vec::new();
        for file in &self.0 {
            for start in (0..file.len()).step_by(CHUNK_BYTES) {
                out.push(Chunk {
                    file,
                    start,
                    end: (start + CHUNK_BYTES).min(file.len()),
                });
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct Chunk<'a> {
    file: &'a [u8],
    start: usize,
    end: usize,
}

impl<'a> Chunk<'a> {
    pub fn rows(self) -> &'a [u8] {
        fn boundary(data: &[u8], off: usize) -> usize {
            let base = off.saturating_sub(1);
            memchr(b'\n', &data[base..]).map_or(data.len(), |i| base + i + 1)
        }
        let a = boundary(self.file, self.start);
        let b = if self.end == self.file.len() {
            self.end
        } else {
            boundary(self.file, self.end)
        };
        &self.file[a..b]
    }
}

fn slow_fields<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    let mut f = [b"".as_slice(); N];
    let mut parts = line.split(|&c| c == b',');
    for value in &mut f {
        let raw = parts.next()?;
        *value = if raw.first() == Some(&b'"') && raw.last() == Some(&b'"') && raw.len() >= 2 {
            &raw[1..raw.len() - 1]
        } else {
            raw
        };
        if memchr(b'"', value).is_some() {
            return None;
        }
    }
    parts.next().is_none().then_some(f)
}

// One newline scan and three AVX2 mask loads handle ordinary rows. The
// padded tail is never assumed readable: short final buffers use the scalar path.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn fast_fields<const N: usize>(remaining: &[u8], n: usize) -> Option<[&[u8]; N]> {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn masks(p: *const u8) -> (u32, u32) {
        unsafe {
            let b = _mm256_loadu_si256(p.cast());
            let comma =
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, _mm256_set1_epi8(b',' as i8))) as u32;
            let quote =
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, _mm256_set1_epi8(b'"' as i8))) as u32;
            (comma, quote)
        }
    }
    if n == 0 || n > 127 || remaining.len() < n.next_multiple_of(32) {
        return slow_fields(&remaining[..n]);
    }
    let mut commas = 0u128;
    let mut quotes = 0u128;
    for pos in (0..n).step_by(32) {
        // pos+32 is bounded by the checked rounded-up line size.
        let (c, q) = unsafe { masks(remaining.as_ptr().add(pos)) };
        commas |= (c as u128) << pos;
        quotes |= (q as u128) << pos;
    }
    let mask = (1u128 << n) - 1;
    commas &= mask;
    quotes &= mask;
    if commas.count_ones() != (N - 1) as u32 {
        return None;
    }
    let mut fields = [b"".as_slice(); N];
    let mut start = 0;
    let mut matched = 0;
    for (idx, f) in fields.iter_mut().enumerate() {
        let end = if idx + 1 == N {
            n
        } else {
            commas.trailing_zeros() as usize
        };
        if idx + 1 != N {
            commas &= commas - 1;
        }
        let raw = &remaining[start..end];
        if raw.first() == Some(&b'"') {
            if raw.len() < 2 || raw.last() != Some(&b'"') {
                return None;
            }
            *f = &raw[1..raw.len() - 1];
            matched += 2;
        } else {
            *f = raw;
        }
        start = end + 1;
    }
    (matched == quotes.count_ones()).then_some(fields)
}

#[inline]
pub fn each<'a, const N: usize>(input: &'a [u8], mut visit: impl FnMut([&'a [u8]; N])) {
    let mut start = 0;
    for end in memchr_iter(b'\n', input).chain(std::iter::once(input.len())) {
        if start > end {
            break;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let fields = fast_fields::<N>(&input[start..], end - start);
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let fields = slow_fields::<N>(&input[start..end]);
        if let Some(f) = fields {
            visit(f)
        }
        start = end + 1;
    }
}

pub fn after_header(input: &[u8]) -> &[u8] {
    memchr(b'\n', input).map_or(&[], |p| &input[p + 1..])
}

pub const DAY: i64 = 86_400_000;
pub const WEEK: i64 = 7 * DAY;

fn digits(s: &[u8]) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    let mut n = 0i64;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(d as i64)?;
    }
    Some(n)
}

pub fn price(s: &[u8]) -> Option<u32> {
    digits(s)
        .filter(|p| (1..=10_000_000).contains(p))
        .map(|x| x as u32)
}

// Most numeric timestamps have thirteen digits. Decode two overlapping
// eight-byte loads without reading beyond the field, including at EOF.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn epoch13(s: &[u8; 13]) -> Option<i64> {
    use std::arch::x86_64::*;
    let first = i64::from_le_bytes(s[..8].try_into().unwrap());
    let last = i64::from_le_bytes(s[5..].try_into().unwrap());
    unsafe {
        let bytes = _mm_set_epi64x(last, first);
        let digits = _mm_sub_epi8(bytes, _mm_set1_epi8(b'0' as i8));
        let invalid = _mm_subs_epu8(digits, _mm_set1_epi8(9));
        if _mm_testz_si128(invalid, invalid) == 0 {
            return None;
        }
        let positions = _mm_setr_epi8(-1, -1, -1, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15);
        let padded = _mm_shuffle_epi8(digits, positions);
        let pairs = _mm_maddubs_epi16(padded, _mm_set1_epi16(0x010a));
        let groups = _mm_madd_epi16(pairs, _mm_set1_epi32(0x00010064));
        let group = |x: i32| i64::from(x);
        let high =
            group(_mm_extract_epi32::<0>(groups)) * 10_000 + group(_mm_extract_epi32::<1>(groups));
        let low =
            group(_mm_extract_epi32::<2>(groups)) * 10_000 + group(_mm_extract_epi32::<3>(groups));
        Some(high * 100_000_000 + low)
    }
}

#[inline]
fn seconds_in_day<const VALIDATE: bool>(clock: &[u8]) -> Option<i64> {
    let word = u64::from_le_bytes(clock.try_into().unwrap());
    if VALIDATE {
        // Check both separators and the high nibble of all six ASCII digits.
        if (word ^ 0x3030_3a30_303a_3030) & 0xf0f0_fff0_f0ff_f0f0 != 0 {
            return None;
        }
        let low = word & 0x0f0f_000f_0f00_0f0f;
        if (low + 0x0606_0606_0606_0606) & 0x1010_1010_1010_1010 != 0 {
            return None;
        }
    }
    const TENS: u64 = 0x000f_0000_0f00_000f;
    let pairs = (word & TENS) * 10 + ((word >> 8) & TENS);
    let (hour, minute, second) = (pairs & 255, (pairs >> 24) & 255, pairs >> 48);
    if VALIDATE && (hour >= 24 || minute >= 60 || second >= 60) {
        return None;
    }
    Some(((hour * 60 + minute) * 60 + second) as i64)
}

fn civil_year(y: i64, m: i64, d: i64) -> Option<i64> {
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let max = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => 28 + i64::from(leap),
        _ => return None,
    };
    if d < 1 || d > max {
        return None;
    }
    let year = y - i64::from(m <= 2);
    let era = year.div_euclid(400);
    let within = year - era * 400;
    let shift = if m > 2 { m - 3 } else { m + 9 };
    Some(
        (era * 146097 + within * 365 + within / 4 - within / 100 + (153 * shift + 2) / 5 + d
            - 1
            - 719468)
            * DAY,
    )
}

pub struct Dates {
    slots: [(u64, u16, i64); 128],
}
impl Dates {
    pub fn new() -> Self {
        Self {
            slots: [(0, 0, i64::MIN); 128],
        }
    }
    fn day(&mut self, s: &[u8]) -> Option<i64> {
        let k = u64::from_ne_bytes(s[..8].try_into().unwrap());
        let d = u16::from_ne_bytes(s[8..10].try_into().unwrap());
        let idx = (((s[6] & 1) << 6) | ((s[8] & 3) << 4) | (s[9] & 15)) as usize;
        let slot = &mut self.slots[idx];
        if slot.0 == k && slot.1 == d && slot.2 != i64::MIN {
            return Some(slot.2);
        }
        if s[4] != b'-' || s[7] != b'-' {
            return None;
        }
        let base = civil_year(digits(&s[..4])?, digits(&s[5..7])?, digits(&s[8..10])?)?;
        *slot = (k, d, base);
        Some(base)
    }
    pub fn ingest(&mut self, s: &[u8]) -> Option<i64> {
        if s.len() != 20 {
            return None;
        }
        Some(self.day(s)? + seconds_in_day::<false>(&s[11..19])? * 1000)
    }
    pub fn time(&mut self, s: &[u8]) -> Option<i64> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Ok(fixed) = s.try_into() {
            return epoch13(fixed);
        }
        if !matches!(s.len(), 20 | 25) || s[4] != b'-' {
            return digits(s);
        }
        if s[10] != b'T' {
            return None;
        }
        let seconds = seconds_in_day::<true>(&s[11..19])?;
        let offset = if s.len() == 20 {
            if s[19] != b'Z' {
                return None;
            }
            0
        } else {
            let (zh, zm) = (digits(&s[20..22])?, digits(&s[23..25])?);
            if zh >= 24 || zm >= 60 || s[22] != b':' {
                return None;
            }
            match s[19] {
                b'+' => zh * 60 + zm,
                b'-' => -(zh * 60 + zm),
                _ => return None,
            }
        };
        Some(self.day(s)? + (seconds - offset * 60) * 1000)
    }
}
