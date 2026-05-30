// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Column-at-a-time SIMD fast path for casting homogeneous RFC3339/ISO8601 string columns to
//! `Timestamp(Nanosecond, UTC)`. A format-homogeneous, fixed-length string column is a contiguous
//! fixed-stride byte buffer, so a 16-record byte-transpose puts digit position `p` across SIMD
//! lanes and the field combine becomes cross-lane multiply-add.
//!
//! This is a conservative fast path: it returns `Some(values)` only when the whole column is
//! null-free, fixed-stride, of a supported UTC format, and every row is valid and in the
//! nanosecond-representable range. Otherwise it returns `None` and the caller uses the general
//! (`chrono`-based) path — so it never changes behaviour, only accelerates the common case.
//!
//! Architecture: format-specific input is pure DATA (`FormatSpec`, built by `spec()`); all logic
//! lives in the generic engine. Adding a format is one line in `FORMATS`.

use arrow_array::{Array, GenericStringArray, OffsetSizeTrait};

/// Declarative description of a fixed-layout timestamp format. Pure data — no per-format logic.
#[derive(Clone, Copy)]
struct FormatSpec {
    width: usize,
    year: [usize; 4],
    mon: [usize; 2],
    day: [usize; 2],
    hour: [usize; 2],
    min: [usize; 2],
    sec: [usize; 2],
    frac: [usize; 9],
    frac_len: usize,
    frac_scale: i64,
    seps: [(usize, u8); 8],
    sep_len: usize,
    needs_tile2: bool,
}

/// Build a `FormatSpec` from the only format-specific input: date/time separator (`T`/space),
/// number of fractional-second digits, and whether a trailing `Z` is present. Standard RFC3339
/// byte positions (date 0-9, time 11-18) are derived.
fn spec(sep: u8, frac_len: usize, tz_z: bool) -> FormatSpec {
    let mut frac = [0usize; 9];
    for i in 0..frac_len {
        frac[i] = 20 + i;
    }
    let mut seps = [(0usize, 0u8); 8];
    seps[0] = (4, b'-');
    seps[1] = (7, b'-');
    seps[2] = (10, sep);
    seps[3] = (13, b':');
    seps[4] = (16, b':');
    let mut sl = 5;
    let mut end = 19;
    if frac_len > 0 {
        seps[sl] = (19, b'.');
        sl += 1;
        end = 20 + frac_len;
    }
    if tz_z {
        seps[sl] = (end, b'Z');
        sl += 1;
        end += 1;
    }
    FormatSpec {
        width: end,
        year: [0, 1, 2, 3],
        mon: [5, 6],
        day: [8, 9],
        hour: [11, 12],
        min: [14, 15],
        sec: [17, 18],
        frac,
        frac_len,
        frac_scale: if frac_len > 0 {
            10i64.pow((9 - frac_len) as u32)
        } else {
            0
        },
        seps,
        sep_len: sl,
        needs_tile2: end >= 25,
    }
}

/// Supported formats, one line each (sep, fractional-digits, trailing-Z). All UTC.
const FORMATS: &[(u8, usize, bool)] = &[
    (b'T', 3, true),
    (b'T', 6, true),
    (b'T', 9, true),
    (b'T', 0, true),
    (b'T', 3, false),
    (b'T', 6, false),
    (b'T', 9, false),
    (b'T', 0, false),
    (b' ', 3, false),
    (b' ', 6, false),
    (b' ', 9, false),
    (b' ', 0, false),
];

/// Detect the column format from one sample record of known width.
fn detect(sample: &[u8]) -> Option<FormatSpec> {
    let w = sample.len();
    for &(sep, fl, tz) in FORMATS {
        let s = spec(sep, fl, tz);
        if s.width == w
            && s.seps[..s.sep_len]
                .iter()
                .all(|&(p, b)| sample.get(p) == Some(&b))
        {
            return Some(s);
        }
    }
    None
}

/// Branchless Hinnant days-from-1970-01-01 (assumes y >= 0, true for parsed years).
#[inline(always)]
fn days_from_civil(y: i32, m: i32, d: i32) -> i32 {
    let y = y - (m <= 2) as i32;
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = m + 9 - 12 * (m > 2) as i32;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Strict calendar validity: month, day-of-month (incl. leap years), and a conservative year
/// range that keeps nanoseconds within i64 (boundary years fall back to the general path).
#[inline(always)]
fn valid(y: u32, m: u32, d: u32, hh: u32, mi: u32, ss: u32) -> bool {
    if !(1..=12).contains(&m) || d < 1 || hh > 23 || mi > 59 || ss > 59 || !(1678..=2261).contains(&y)
    {
        return false;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dim = match m {
        2 => 28 + leap as u32,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    d <= dim
}

const NANOS_PER_SEC: i64 = 1_000_000_000;
const NANOS_PER_DAY: i64 = 86_400 * NANOS_PER_SEC;

/// Strict scalar parse of one fixed-format record -> nanos UTC, or None if invalid (caller bails).
#[inline]
fn parse_one(b: &[u8], spec: &FormatSpec) -> Option<i64> {
    for s in 0..spec.sep_len {
        let (p, c) = spec.seps[s];
        if b.get(p) != Some(&c) {
            return None;
        }
    }
    let rd = |pos: &[usize]| -> Option<u32> {
        let mut v = 0u32;
        for &p in pos {
            let dgt = b[p].wrapping_sub(b'0');
            if dgt >= 10 {
                return None;
            }
            v = v * 10 + dgt as u32;
        }
        Some(v)
    };
    let (year, mon, day) = (rd(&spec.year)?, rd(&spec.mon)?, rd(&spec.day)?);
    let (hh, mm, ss) = (rd(&spec.hour)?, rd(&spec.min)?, rd(&spec.sec)?);
    let frac = if spec.frac_len > 0 {
        rd(&spec.frac[..spec.frac_len])?
    } else {
        0
    };
    if !valid(year, mon, day, hh, mm, ss) {
        return None;
    }
    let days = days_from_civil(year as i32, mon as i32, day as i32) as i64;
    Some(days * NANOS_PER_DAY
        + (hh as i64 * 3600 + mm as i64 * 60 + ss as i64) * NANOS_PER_SEC
        + frac as i64 * spec.frac_scale)
}

/// Try the SIMD fast path on a generic string array. Returns `Some(nanos)` only if the whole
/// column was handled (null-free, fixed-stride, supported format, all rows valid); else `None`.
///
/// The architecture-neutral prologue (null / fixed-stride / format checks) is shared; the
/// per-record kernel is dispatched by `run_simd_kernel` — NEON on aarch64, AVX2 on x86_64
/// (runtime-detected), and no fast path on other targets.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn try_cast_to_timestamp_nanos<O: OffsetSizeTrait>(
    array: &GenericStringArray<O>,
) -> Option<Vec<i64>> {
    if array.null_count() != 0 {
        return None;
    }
    let n = array.len();
    if n == 0 {
        return None;
    }
    let offsets = array.value_offsets();
    let base = offsets[0].as_usize();
    let w = offsets[1].as_usize() - base;
    if w < 19 || w > 31 {
        return None;
    }
    // require a fixed stride (offsets[i] == base + i*w) — true iff all strings have length w
    for i in 0..=n {
        if offsets[i].as_usize() != base + i * w {
            return None;
        }
    }
    let data = &array.value_data()[base..base + n * w];
    let spec = detect(&data[0..w])?;
    let mut out = vec![0i64; n];
    if run_simd_kernel(&spec, data, n, &mut out) {
        Some(out)
    } else {
        None
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn try_cast_to_timestamp_nanos<O: OffsetSizeTrait>(
    _array: &GenericStringArray<O>,
) -> Option<Vec<i64>> {
    None // no SIMD kernel for this architecture; caller uses the general (chrono) path
}

/// NEON kernel dispatch (aarch64). NEON is a baseline feature, so it is always available.
#[cfg(target_arch = "aarch64")]
#[inline]
fn run_simd_kernel(spec: &FormatSpec, data: &[u8], n: usize, out: &mut [i64]) -> bool {
    // SAFETY: `data` has n*w bytes; the NEON engine only reads within it (see `full` bound).
    unsafe { neon::parse_all(spec, data, n, out) }
}

/// AVX2 kernel dispatch (x86_64). Gated on runtime detection so the binary stays portable across
/// x86_64 CPUs; without AVX2 we decline and the caller uses the general path.
#[cfg(target_arch = "x86_64")]
#[inline]
fn run_simd_kernel(spec: &FormatSpec, data: &[u8], n: usize, out: &mut [i64]) -> bool {
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available (checked above); `data` has n*w bytes and the AVX2 engine
        // only reads within it (same `safe`/`full` bound as the NEON engine).
        unsafe { avx2::parse_all(spec, data, n, out) }
    } else {
        false
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    // This module is entirely NEON intrinsics; treat unsafe-fn bodies as unsafe blocks.
    // (A PR would instead use explicit `unsafe {}` blocks per arrow-rs style.)
    #![allow(unsafe_op_in_unsafe_fn)]
    use super::{days_from_civil, parse_one, valid, FormatSpec, NANOS_PER_DAY, NANOS_PER_SEC};
    use core::arch::aarch64::*;

    /// 16x16 byte transpose: v[i]=row i in -> v[j]=column j out (recursive TRN butterfly).
    #[inline(always)]
    unsafe fn transpose16(v: &mut [uint8x16_t; 16]) {
        let mut i = 0;
        while i < 16 {
            let a = vtrn1q_u8(v[i], v[i + 1]);
            let b = vtrn2q_u8(v[i], v[i + 1]);
            v[i] = a;
            v[i + 1] = b;
            i += 2;
        }
        i = 0;
        while i < 16 {
            for k in 0..2 {
                let (x, y) = (i + k, i + k + 2);
                let a = vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(v[x]), vreinterpretq_u16_u8(v[y])));
                let b = vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(v[x]), vreinterpretq_u16_u8(v[y])));
                v[x] = a;
                v[y] = b;
            }
            i += 4;
        }
        i = 0;
        while i < 16 {
            for k in 0..4 {
                let (x, y) = (i + k, i + k + 4);
                let a = vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(v[x]), vreinterpretq_u32_u8(v[y])));
                let b = vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(v[x]), vreinterpretq_u32_u8(v[y])));
                v[x] = a;
                v[y] = b;
            }
            i += 8;
        }
        for k in 0..8 {
            let (x, y) = (k, k + 8);
            let a = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(v[x]), vreinterpretq_u64_u8(v[y])));
            let b = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(v[x]), vreinterpretq_u64_u8(v[y])));
            v[x] = a;
            v[y] = b;
        }
    }

    #[inline(always)]
    unsafe fn comb16(pbuf: &[[u8; 16]; 32], positions: &[usize], z: uint8x16_t) -> [u16; 16] {
        let (mut lo, mut hi, k) = (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(10));
        for &p in positions {
            let d = vsubq_u8(vld1q_u8(pbuf[p].as_ptr()), z);
            lo = vaddq_u16(vmulq_u16(lo, k), vmovl_u8(vget_low_u8(d)));
            hi = vaddq_u16(vmulq_u16(hi, k), vmovl_u8(vget_high_u8(d)));
        }
        let mut o = [0u16; 16];
        vst1q_u16(o.as_mut_ptr(), lo);
        vst1q_u16(o[8..].as_mut_ptr(), hi);
        o
    }
    #[inline(always)]
    unsafe fn comb32(pbuf: &[[u8; 16]; 32], positions: &[usize], z: uint8x16_t) -> [u32; 16] {
        let mut a = [vdupq_n_u32(0); 4];
        let k = vdupq_n_u32(10);
        for &p in positions {
            let d8 = vsubq_u8(vld1q_u8(pbuf[p].as_ptr()), z);
            let l = vmovl_u8(vget_low_u8(d8));
            let h = vmovl_u8(vget_high_u8(d8));
            let d = [
                vmovl_u16(vget_low_u16(l)),
                vmovl_u16(vget_high_u16(l)),
                vmovl_u16(vget_low_u16(h)),
                vmovl_u16(vget_high_u16(h)),
            ];
            for i in 0..4 {
                a[i] = vaddq_u32(vmulq_u32(a[i], k), d[i]);
            }
        }
        let mut o = [0u32; 16];
        for i in 0..4 {
            vst1q_u32(o[i * 4..].as_mut_ptr(), a[i]);
        }
        o
    }

    /// Generic engine: parse n records from a fixed-stride buffer. Returns false on the first
    /// invalid row (caller -> None -> general path). Bounds tile loads to stay within `data`.
    pub(super) unsafe fn parse_all(spec: &FormatSpec, data: &[u8], n: usize, out: &mut [i64]) -> bool {
        let (w, base, z) = (spec.width, data.as_ptr(), vdupq_n_u8(b'0'));
        let nine = vdupq_n_u8(9);
        // last record whose tile reads (up to start+32) stay within `data`
        let safe = if data.len() >= 32 { (data.len() - 32) / w + 1 } else { 0 };
        let full = (safe.min(n) / 16) * 16;
        let mut pbuf = [[0u8; 16]; 32];
        let mut j = 0;
        while j < full {
            let mut t = [vdupq_n_u8(0); 16];
            for r in 0..16 {
                t[r] = vld1q_u8(base.add((j + r) * w));
            }
            transpose16(&mut t);
            for i in 0..16 {
                vst1q_u8(pbuf[i].as_mut_ptr(), t[i]);
            }
            for r in 0..16 {
                t[r] = vld1q_u8(base.add((j + r) * w + 8));
            }
            transpose16(&mut t);
            for i in 0..16 {
                vst1q_u8(pbuf[8 + i].as_mut_ptr(), t[i]);
            }
            if spec.needs_tile2 {
                for r in 0..16 {
                    t[r] = vld1q_u8(base.add((j + r) * w + 16));
                }
                transpose16(&mut t);
                for i in 0..16 {
                    vst1q_u8(pbuf[16 + i].as_mut_ptr(), t[i]);
                }
            }
            // SIMD validity: separators correct + field positions are ASCII digits
            let mut ok = vdupq_n_u8(0xFF);
            for s in 0..spec.sep_len {
                let (p, b) = spec.seps[s];
                ok = vandq_u8(ok, vceqq_u8(vld1q_u8(pbuf[p].as_ptr()), vdupq_n_u8(b)));
            }
            for grp in [
                &spec.year[..],
                &spec.mon[..],
                &spec.day[..],
                &spec.hour[..],
                &spec.min[..],
                &spec.sec[..],
                &spec.frac[..spec.frac_len],
            ] {
                for &p in grp {
                    let d = vsubq_u8(vld1q_u8(pbuf[p].as_ptr()), z);
                    ok = vandq_u8(ok, vcleq_u8(d, nine));
                }
            }
            if vminvq_u8(ok) != 0xFF {
                return false;
            }
            let year = comb16(&pbuf, &spec.year, z);
            let mon = comb16(&pbuf, &spec.mon, z);
            let day = comb16(&pbuf, &spec.day, z);
            let hour = comb16(&pbuf, &spec.hour, z);
            let min = comb16(&pbuf, &spec.min, z);
            let sec = comb16(&pbuf, &spec.sec, z);
            let frac = if spec.frac_len > 0 {
                comb32(&pbuf, &spec.frac[..spec.frac_len], z)
            } else {
                [0u32; 16]
            };
            // strict scalar calendar validation + nanos (date math vectorizes via branchless i32)
            for r in 0..16 {
                let (y, m, d) = (year[r] as u32, mon[r] as u32, day[r] as u32);
                if !valid(y, m, d, hour[r] as u32, min[r] as u32, sec[r] as u32) {
                    return false;
                }
                let days = days_from_civil(y as i32, m as i32, d as i32) as i64;
                out[j + r] = days * NANOS_PER_DAY
                    + (hour[r] as i64 * 3600 + min[r] as i64 * 60 + sec[r] as i64) * NANOS_PER_SEC
                    + frac[r] as i64 * spec.frac_scale;
            }
            j += 16;
        }
        while j < n {
            match parse_one(&data[j * w..j * w + w], spec) {
                Some(v) => out[j] = v,
                None => return false,
            }
            j += 1;
        }
        true
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    // This module is entirely AVX2 intrinsics; treat unsafe-fn bodies as unsafe blocks.
    // (A PR would instead use explicit `unsafe {}` blocks per arrow-rs style.) Every function is
    // `#[target_feature(enable = "avx2")]` and only reached after a runtime AVX2 check in the
    // dispatcher, so the intrinsics are always legal at the point of call.
    #![allow(unsafe_op_in_unsafe_fn)]
    use super::{days_from_civil, parse_one, valid, FormatSpec, NANOS_PER_DAY, NANOS_PER_SEC};
    use core::arch::x86_64::*;

    /// Output-lane permutation of `transpose16`. x86 has no byte-granularity de-interleave (the
    /// NEON `TRN` butterfly), so we build the transpose from `unpack` (interleave) instead. That
    /// ladder lands position `p` in register `BITREV[.]`-reversed order, so the store step writes
    /// register `k` to logical position `BITREV[k]`. `BITREV[k]` = bit-reversal of `k` in 4 bits.
    const BITREV: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

    /// 16x16 byte transpose: v[i]=row i in -> (bit-reversed) v[k]=column `BITREV[k]` out. Four
    /// `unpack` stages at 8/16/32/64-bit granularity, pairing registers at distance 1/2/4/8.
    #[target_feature(enable = "avx2")]
    unsafe fn transpose16(v: &mut [__m128i; 16]) {
        let mut i = 0;
        while i < 16 {
            let a = _mm_unpacklo_epi8(v[i], v[i + 1]);
            let b = _mm_unpackhi_epi8(v[i], v[i + 1]);
            v[i] = a;
            v[i + 1] = b;
            i += 2;
        }
        i = 0;
        while i < 16 {
            for k in 0..2 {
                let (x, y) = (i + k, i + k + 2);
                let a = _mm_unpacklo_epi16(v[x], v[y]);
                let b = _mm_unpackhi_epi16(v[x], v[y]);
                v[x] = a;
                v[y] = b;
            }
            i += 4;
        }
        i = 0;
        while i < 16 {
            for k in 0..4 {
                let (x, y) = (i + k, i + k + 4);
                let a = _mm_unpacklo_epi32(v[x], v[y]);
                let b = _mm_unpackhi_epi32(v[x], v[y]);
                v[x] = a;
                v[y] = b;
            }
            i += 8;
        }
        for k in 0..8 {
            let (x, y) = (k, k + 8);
            let a = _mm_unpacklo_epi64(v[x], v[y]);
            let b = _mm_unpackhi_epi64(v[x], v[y]);
            v[x] = a;
            v[y] = b;
        }
    }

    /// Store the 16 transposed registers for the tile starting at byte `off` into `pbuf`, undoing
    /// the bit-reversal so `pbuf[off + p]` holds byte position `off + p` across the 16 records.
    #[target_feature(enable = "avx2")]
    unsafe fn store_tile(t: &[__m128i; 16], pbuf: &mut [[u8; 16]; 32], off: usize) {
        for k in 0..16 {
            _mm_storeu_si128(pbuf[off + BITREV[k]].as_mut_ptr() as *mut __m128i, t[k]);
        }
    }

    /// Combine the digit positions in `positions` into a per-record value, ≤ 16 bits wide (year,
    /// month, …). 16 records run in parallel across the u16 lanes of one 256-bit register.
    #[target_feature(enable = "avx2")]
    unsafe fn comb16(pbuf: &[[u8; 16]; 32], positions: &[usize], zero: __m128i) -> [u16; 16] {
        let ten = _mm256_set1_epi16(10);
        let mut acc = _mm256_setzero_si256();
        for &p in positions {
            let d = _mm_sub_epi8(_mm_loadu_si128(pbuf[p].as_ptr() as *const __m128i), zero);
            let d16 = _mm256_cvtepu8_epi16(d);
            acc = _mm256_add_epi16(_mm256_mullo_epi16(acc, ten), d16);
        }
        let mut o = [0u16; 16];
        _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, acc);
        o
    }

    /// Combine the digit positions in `positions` into a per-record value up to 32 bits wide
    /// (fractional seconds, up to 9 digits). 16 records span two 256-bit u32 registers.
    #[target_feature(enable = "avx2")]
    unsafe fn comb32(pbuf: &[[u8; 16]; 32], positions: &[usize], zero: __m128i) -> [u32; 16] {
        let ten = _mm256_set1_epi32(10);
        let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for &p in positions {
            let d = _mm_sub_epi8(_mm_loadu_si128(pbuf[p].as_ptr() as *const __m128i), zero);
            let dlo = _mm256_cvtepu8_epi32(d);
            let dhi = _mm256_cvtepu8_epi32(_mm_srli_si128(d, 8));
            lo = _mm256_add_epi32(_mm256_mullo_epi32(lo, ten), dlo);
            hi = _mm256_add_epi32(_mm256_mullo_epi32(hi, ten), dhi);
        }
        let mut o = [0u32; 16];
        _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, lo);
        _mm256_storeu_si256(o[8..].as_mut_ptr() as *mut __m256i, hi);
        o
    }

    /// Generic engine: parse n records from a fixed-stride buffer. Returns false on the first
    /// invalid row (caller -> None -> general path). Bounds tile loads to stay within `data`.
    pub(super) unsafe fn parse_all(spec: &FormatSpec, data: &[u8], n: usize, out: &mut [i64]) -> bool {
        let (w, base, zero) = (spec.width, data.as_ptr(), _mm_set1_epi8(b'0' as i8));
        let nine = _mm_set1_epi8(9);
        // last record whose tile reads (up to start+32) stay within `data`
        let safe = if data.len() >= 32 { (data.len() - 32) / w + 1 } else { 0 };
        let full = (safe.min(n) / 16) * 16;
        let mut pbuf = [[0u8; 16]; 32];
        let mut j = 0;
        while j < full {
            let mut t = [_mm_setzero_si128(); 16];
            for r in 0..16 {
                t[r] = _mm_loadu_si128(base.add((j + r) * w) as *const __m128i);
            }
            transpose16(&mut t);
            store_tile(&t, &mut pbuf, 0);
            for r in 0..16 {
                t[r] = _mm_loadu_si128(base.add((j + r) * w + 8) as *const __m128i);
            }
            transpose16(&mut t);
            store_tile(&t, &mut pbuf, 8);
            if spec.needs_tile2 {
                for r in 0..16 {
                    t[r] = _mm_loadu_si128(base.add((j + r) * w + 16) as *const __m128i);
                }
                transpose16(&mut t);
                store_tile(&t, &mut pbuf, 16);
            }
            // SIMD validity: separators correct + field positions are ASCII digits
            let mut ok = _mm_set1_epi8(-1); // 0xFF in every lane
            for s in 0..spec.sep_len {
                let (p, b) = spec.seps[s];
                let eq = _mm_cmpeq_epi8(
                    _mm_loadu_si128(pbuf[p].as_ptr() as *const __m128i),
                    _mm_set1_epi8(b as i8),
                );
                ok = _mm_and_si128(ok, eq);
            }
            for grp in [
                &spec.year[..],
                &spec.mon[..],
                &spec.day[..],
                &spec.hour[..],
                &spec.min[..],
                &spec.sec[..],
                &spec.frac[..spec.frac_len],
            ] {
                for &p in grp {
                    let d = _mm_sub_epi8(_mm_loadu_si128(pbuf[p].as_ptr() as *const __m128i), zero);
                    // (raw - '0') <= 9 unsigned  <=>  min_epu8(d, 9) == d
                    let le = _mm_cmpeq_epi8(_mm_min_epu8(d, nine), d);
                    ok = _mm_and_si128(ok, le);
                }
            }
            if _mm_movemask_epi8(ok) != 0xFFFF {
                return false;
            }
            let year = comb16(&pbuf, &spec.year, zero);
            let mon = comb16(&pbuf, &spec.mon, zero);
            let day = comb16(&pbuf, &spec.day, zero);
            let hour = comb16(&pbuf, &spec.hour, zero);
            let min = comb16(&pbuf, &spec.min, zero);
            let sec = comb16(&pbuf, &spec.sec, zero);
            let frac = if spec.frac_len > 0 {
                comb32(&pbuf, &spec.frac[..spec.frac_len], zero)
            } else {
                [0u32; 16]
            };
            // strict scalar calendar validation + nanos (date math vectorizes via branchless i32)
            for r in 0..16 {
                let (y, m, d) = (year[r] as u32, mon[r] as u32, day[r] as u32);
                if !valid(y, m, d, hour[r] as u32, min[r] as u32, sec[r] as u32) {
                    return false;
                }
                let days = days_from_civil(y as i32, m as i32, d as i32) as i64;
                out[j + r] = days * NANOS_PER_DAY
                    + (hour[r] as i64 * 3600 + min[r] as i64 * 60 + sec[r] as i64) * NANOS_PER_SEC
                    + frac[r] as i64 * spec.frac_scale;
            }
            j += 16;
        }
        while j < n {
            match parse_one(&data[j * w..j * w + w], spec) {
                Some(v) => out[j] = v,
                None => return false,
            }
            j += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::string_to_timestamp_nanos;
    use arrow_array::StringArray;

    fn mk(fmt: &str, n: usize) -> Vec<String> {
        let mut v = Vec::with_capacity(n);
        let mut x: u64 = 0x1234567;
        for _ in 0..n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let yr = 1970 + (x >> 32) % 280;
            let mo = 1 + (x >> 20) % 12;
            let da = 1 + (x >> 8) % 28;
            let hh = (x >> 40) % 24;
            let mi = (x >> 16) % 60;
            let se = (x >> 4) % 60;
            let us = (x >> 24) % 1_000_000;
            let ms = us / 1000;
            v.push(match fmt {
                "z_ms" => format!("{yr:04}-{mo:02}-{da:02}T{hh:02}:{mi:02}:{se:02}.{ms:03}Z"),
                "z_us" => format!("{yr:04}-{mo:02}-{da:02}T{hh:02}:{mi:02}:{se:02}.{us:06}Z"),
                "no_tz" => format!("{yr:04}-{mo:02}-{da:02}T{hh:02}:{mi:02}:{se:02}"),
                "space" => format!("{yr:04}-{mo:02}-{da:02} {hh:02}:{mi:02}:{se:02}.{ms:03}"),
                _ => unreachable!(),
            });
        }
        v
    }

    /// Whether a SIMD kernel is active for the current target: NEON is baseline on aarch64, AVX2
    /// is runtime-detected on x86_64, and there is no kernel elsewhere. Mirrors `run_simd_kernel`.
    fn simd_available() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[test]
    fn fast_path_matches_general() {
        for fmt in ["z_ms", "z_us", "no_tz", "space"] {
            let corpus = mk(fmt, 5000);
            let array = StringArray::from(corpus.clone());
            let fast = try_cast_to_timestamp_nanos(&array);
            // With a SIMD kernel: values must be identical to the general parser. Otherwise: None.
            if simd_available() {
                let fast = fast.unwrap_or_else(|| panic!("{fmt}: fast path declined a valid column"));
                for (i, s) in corpus.iter().enumerate() {
                    assert_eq!(fast[i], string_to_timestamp_nanos(s).unwrap(), "{fmt} row {i}: {s}");
                }
            } else {
                assert!(fast.is_none());
            }
        }
    }

    #[test]
    fn invalid_dates_decline_fast_path() {
        // Feb 30 is well-formed but not a real date -> the general parser errors, so the fast
        // path must decline (None) rather than silently produce a value.
        let array = StringArray::from(vec![
            "2020-01-15T00:00:00.000Z".to_string(),
            "2020-02-30T00:00:00.000Z".to_string(), // invalid day
        ]);
        assert!(try_cast_to_timestamp_nanos(&array).is_none());
        assert!(string_to_timestamp_nanos("2020-02-30T00:00:00.000Z").is_err());
    }

    #[test]
    fn nulls_and_mixed_width_decline() {
        // null present -> decline
        let with_null = StringArray::from(vec![
            Some("2020-01-15T00:00:00.000Z".to_string()),
            None,
        ]);
        assert!(try_cast_to_timestamp_nanos(&with_null).is_none());
        // non-uniform stride (different lengths) -> decline
        let mixed = StringArray::from(vec![
            "2020-01-15T00:00:00.000Z".to_string(),
            "2020-01-15T00:00:00Z".to_string(),
        ]);
        assert!(try_cast_to_timestamp_nanos(&mixed).is_none());
    }

    #[test]
    fn wrong_separator_declines() {
        // same width, valid digits, but '/' instead of '-' is not a supported format
        let array = StringArray::from(vec![
            "2020/01/15T00:00:00.000Z".to_string(),
            "2020/01/16T00:00:00.000Z".to_string(),
        ]);
        assert!(try_cast_to_timestamp_nanos(&array).is_none());
    }
}
