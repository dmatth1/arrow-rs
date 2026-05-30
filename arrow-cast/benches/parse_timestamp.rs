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

use arrow_cast::parse::string_to_timestamp_nanos;
use criterion::*;
use std::hint;

// Format variants ordered by complexity:
//   bare date, no-tz, tz-Z, tz-offset, fractional variants
const TIMESTAMPS: &[(&str, &str)] = &[
    ("date_only",           "2020-09-08"),
    ("no_frac_no_tz",       "2020-09-08T13:42:29"),
    ("frac_ms_no_tz",       "2020-09-08T13:42:29.190"),
    ("frac_us_no_tz",       "2020-09-08T13:42:29.190855"),
    ("frac_ns_no_tz",       "2020-09-08T13:42:29.190855999"),
    ("no_frac_tz_z",        "2020-09-08T13:42:29Z"),
    ("no_frac_tz_offset",   "2020-09-08T13:42:29+00:00"),
    ("frac_ms_tz_offset",   "2020-09-08T13:42:29.190+00:00"),
    ("frac_us_tz_offset",   "2020-09-08T13:42:29.190855+00:00"),
    ("frac_ns_tz_neg",      "2020-09-08T13:42:29.190855999-05:00"),
    ("frac_us_tz_z",        "2020-09-08T13:42:29.190855Z"),
    ("space_sep_no_tz",     "2020-09-08 13:42:29.190855"),
];

/// Per-format single-call bench (existing shape, keeps CI history).
fn bench_single(c: &mut Criterion) {
    for (name, ts) in TIMESTAMPS {
        let t = hint::black_box(*ts);
        c.bench_function(&format!("single/{name}"), |b| {
            b.iter(|| string_to_timestamp_nanos(t).unwrap());
        });
    }
}

/// Bulk throughput bench: 10k pre-interleaved timestamps per Criterion iteration,
/// reported as throughput in elements/s so Criterion prints GB/s and ns/elem.
fn bench_bulk(c: &mut Criterion) {
    // Build a 10k batch interleaving all format variants — prevents branch
    // predictor from learning a single format.
    const BATCH: usize = 10_000;
    let corpus: Vec<&str> = (0..BATCH)
        .map(|i| TIMESTAMPS[i % TIMESTAMPS.len()].1)
        .collect();

    let mut group = c.benchmark_group("bulk");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("mixed_formats", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for ts in hint::black_box(corpus.as_slice()) {
                sum = sum.wrapping_add(string_to_timestamp_nanos(ts).unwrap());
            }
            hint::black_box(sum)
        });
    });
    group.finish();

    // Per-format bulk groups (pure same-format batch) — measures the hot-path
    // cost when a column is homogeneous (typical real CSV).
    let mut group = c.benchmark_group("bulk_homogeneous");
    group.throughput(Throughput::Elements(BATCH as u64));
    for (name, ts) in TIMESTAMPS {
        let batch: Vec<&str> = std::iter::repeat(*ts).take(BATCH).collect();
        group.bench_function(*name, |b| {
            b.iter(|| {
                let mut sum: i64 = 0;
                for t in hint::black_box(batch.as_slice()) {
                    sum = sum.wrapping_add(string_to_timestamp_nanos(t).unwrap());
                }
                hint::black_box(sum)
            });
        });
    }
    group.finish();
}

/// Bulk `cast(StringArray -> Timestamp(Nanosecond))` throughput via the public API. A
/// homogeneous, null-free, fixed-length column hits the SIMD fast path; a null-containing column
/// of the same data declines to the general (chrono) path — so the two functions together show
/// the fast-path speedup on the same workload.
fn bench_cast(c: &mut Criterion) {
    use arrow_array::StringArray;
    use arrow_schema::{DataType, TimeUnit};

    const N: usize = 8192;
    let to = DataType::Timestamp(TimeUnit::Nanosecond, None);

    // varied but same-format ("YYYY-MM-DDTHH:MM:SS.mmmZ", width 24) -> fixed stride -> fast path
    let mut x: u64 = 0x1234567;
    let strings: Vec<String> = (0..N)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let yr = 2000 + (x >> 32) % 26;
            let mo = 1 + (x >> 20) % 12;
            let da = 1 + (x >> 8) % 28;
            let hh = (x >> 40) % 24;
            let mi = (x >> 16) % 60;
            let se = (x >> 4) % 60;
            let ms = (x >> 24) % 1000;
            format!("{yr:04}-{mo:02}-{da:02}T{hh:02}:{mi:02}:{se:02}.{ms:03}Z")
        })
        .collect();
    let homogeneous = StringArray::from(strings.clone());

    // one null forces the whole array onto the general path (same data -> baseline cost)
    let mut opt: Vec<Option<String>> = strings.into_iter().map(Some).collect();
    opt[0] = None;
    let fallback = StringArray::from(opt);

    let mut group = c.benchmark_group("cast_string_to_timestamp_ns");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("homogeneous_simd_fastpath", |b| {
        b.iter(|| arrow_cast::cast(hint::black_box(&homogeneous), &to).unwrap())
    });
    group.bench_function("general_path_baseline", |b| {
        b.iter(|| arrow_cast::cast(hint::black_box(&fallback), &to).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_single, bench_bulk, bench_cast);
criterion_main!(benches);
