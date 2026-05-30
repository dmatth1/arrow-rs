# AVX2 timestamp-cast fast path — benchmark results

Measured speedup of the SIMD fast path for
`cast(Utf8 -> Timestamp(Nanosecond, UTC))` added in this branch, on x86_64 with
the AVX2 kernel active.

## Environment

- CPU: Intel(R) Xeon(R) @ 2.80GHz (AVX2 + AVX512F available)
- rustc 1.91.1, `RUSTFLAGS="-C target-cpu=native"`
- Benchmark: `arrow-cast/benches/parse_timestamp.rs`, group
  `cast_string_to_timestamp_ns` (8192 homogeneous RFC3339 nanosecond strings,
  format `YYYY-MM-DDTHH:MM:SS.mmmZ`, width 24)

Run with:

```
RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p arrow-cast --bench parse_timestamp -- cast_string_to_timestamp_ns
```

The two arms run the *same* data: `homogeneous_simd_fastpath` is a null-free,
fixed-stride column that hits the AVX2 kernel; `general_path_baseline` is the
same column with one null, which forces the general (`chrono`) path.

## Results

| Path                      | Time (median) | Throughput      |
| ------------------------- | ------------- | --------------- |
| `homogeneous_simd_fastpath` (AVX2) | 187.94 µs | 43.59 Melem/s |
| `general_path_baseline` (chrono)   | 437.16 µs | 18.74 Melem/s |

**Speedup: ~2.33x** (43.59 / 18.74 Melem/s) over the general chrono path.

For reference, before the AVX2 kernel existed the x86_64 path was a no-op
fallback to chrono, so `homogeneous_simd_fastpath` measured ~439 µs — identical
to the baseline. Enabling AVX2 dropped it to ~188 µs, which Criterion reports as
a 57% time reduction / +133% throughput for that benchmark:

```
cast_string_to_timestamp_ns/homogeneous_simd_fastpath
                        time:   [187.52 µs 187.94 µs 188.40 µs]
                        thrpt:  [43.481 Melem/s 43.590 Melem/s 43.686 Melem/s]
                 change:
                        time:   [-57.359% -57.151% -56.927%] (p = 0.00 < 0.05)
                        thrpt:  [+132.16% +133.38% +134.52%]
                        Performance has improved.

cast_string_to_timestamp_ns/general_path_baseline
                        time:   [436.15 µs 437.16 µs 438.82 µs]
                        thrpt:  [18.668 Melem/s 18.739 Melem/s 18.783 Melem/s]
                 change:  Change within noise threshold.
```

## Notes

- The fast path engages only for null-free, fixed-width, homogeneous UTC
  columns (the common CSV/Parquet case). Mixed-width, null-containing, or
  offset/non-UTC columns still take the general path — behaviour is identical,
  only accelerated.
- Correctness is covered by `timestamp_simd::tests::fast_path_matches_general`,
  which checks 20,000 AVX2-produced values (4 formats x 5000 rows) byte-for-byte
  against the general parser.
