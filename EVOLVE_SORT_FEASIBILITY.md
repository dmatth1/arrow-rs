# Can the evolve-sort primitives improve arrow-rs?

> Feasibility investigation, 2026-05-28. Grounds the `evolve` repo's
> `sort/NOVELTY_BRAINSTORM.md` thesis against the *actual* arrow-rs code.
> All evolve performance numbers below were measured in the evolve repo on
> M1/x86 against numpy/std::sort/VQSort/rantala — **not yet on Arrow data**.
> Treat them as "what the primitive does elsewhere," i.e. an upper bound on
> the plausible Arrow win, pending a real benchmark. Status: **investigation,
> no code committed yet.**

## TL;DR

| Opportunity | Where | Novelty | Confidence | Risk | Verdict |
|---|---|---|---|---|---|
| **Radix argsort for primitive numeric columns** | `arrow-ord` `sort_primitive` | low (ips2ra/numpy do it) | **high** | **low** | **Do this first** |
| **Herf float radix** (f32/f64 argsort) | same path | low (Herf 2001) | high | low | Same PR, free rider |
| **Deferred-materialization row sort** | `arrow-row` + `arrow-ord` lexsort | **med–high (the publishable bet)** | low | **high** | Prototype-only; partly pre-empted |
| Radix over materialized row bytes | `arrow-ord` lexsort | low | med | med | Maybe; rows are var-length |

The honest headline: the **low-risk, high-confidence** win is dropping a radix
argsort into the single-column primitive path — Arrow sorts numeric columns
with a *comparison* sort today, exactly numpy's weak spot. The **novel**
idea from the brainstorm (deferred materialization) is real but high-risk and
**partially pre-empted** by prefix short-circuiting Arrow already ships.

---

## What arrow-rs actually does today (verified)

### There are two multi-column sort paths

1. **`LexicographicalComparator`** (`arrow-ord/src/sort.rs:1016`) — per-column
   `DynComparator`s, compared on demand, no row encoding. Default for >5
   columns; a fixed-N specialization handles 2–5 (`sort.rs:939–988`).
2. **Row format** (`arrow-row`) — opt-in. The code itself flags it
   (`arrow-ord/src/sort.rs:914`: *"for multi-column sorts without a limit,
   using the row format may be significantly faster"*). It is **not** the
   default.

### Single primitive columns are comparison-sorted (the opportunity)

`sort_to_indices` → `sort_primitive` (`arrow-ord/src/sort.rs:331`):

```rust
fn sort_primitive<T: ArrowPrimitiveType>(...) -> UInt32Array {
    let mut valids = value_indices
        .into_iter()
        .map(|index| (index, values.value(index as usize)))
        .collect::<Vec<(u32, T::Native)>>();
    sort_impl(options, &mut valids, &nulls, limit, T::Native::compare).into()
}
```

This builds `(index, value)` pairs and `sort_unstable_by`s them with a
comparator (`sort.rs:342`, `630`). That is **precisely** the indirect
comparison argsort that the evolve repo beats by 4–20× with a radix co-sort.
There is **no radix path** for primitive columns anywhere in `arrow-ord`.

### Strings already do a 4-byte prefix short-circuit

`sort_bytes` (`arrow-ord/src/sort.rs:345`) extracts a big-endian 4-byte prefix
into `(idx, prefix:u32, len)` and compares the prefix first, only falling
back to a full `memcmp` on prefix ties (`sort.rs:352–409`). ByteView arrays do
the same in `arrow-ord/src/cmp.rs:740–840` (inline ≤12B key, then 4-byte
prefix, then full slice). **This is the comparison-time half of the evolve
"cheap prefix, resolve ties deeper" idea — already in the tree.**

### The row format fully materializes every byte (confirmed)

`RowConverter::append` (`arrow-row/src/lib.rs:919–1019`):
`row_lengths()` pre-computes the exact encoded length of **every** row
(`lib.rs:1674`), `rows.buffer.resize(total, 0)` pre-allocates the whole
buffer, then `encode_column` writes **all** bytes for **all** rows. Encoding is
eager and complete:

- ints: 1 null/valid marker + big-endian payload, sign-bit flip for signed
  (`lib.rs:200–241`) — *the same order-preserving encoding the evolve repo uses*.
- floats: IEEE→ordered-int bit flip (`lib.rs:243`) — *this is exactly the Herf
  transform the evolve `f64_radix` uses; Arrow already has it.*
- strings: block-based padded COBS-like scheme (`arrow-row/src/variable.rs`),
  2–3× space amplification on string-heavy data.

Sorting then compares rows as plain `&[u8]` via `memcmp` (`Row::cmp`,
`lib.rs:1433`). So the row path is **encode-everything, then comparison-sort
the encoded keys** — never a radix, never deferred.

---

## Where the evolve primitives map in

### 1. Radix argsort for primitive numeric columns — **recommended first PR**

**Target:** `sort_primitive` / `sort_to_indices` (`arrow-ord/src/sort.rs:271,331`).

**Primitive:** evolve `unified/argsort.hpp` + `sort_integers.hpp` — LSD radix
co-sorting `(value, index)`, with:
- counting-collapse when `range ≤ 4n`,
- constant-digit skipping,
- range-normalized passes (`ceil(bitlen(range)/11)` passes),
- adaptive prescan that no-ops sorted/reverse input.

**Why it fits:** Arrow's input is already a contiguous `PrimitiveArray<T>` of
fixed-width native ints — the ideal radix substrate. Output is `UInt32Array`
indices, which is just the index half of the co-sort. The existing comparison
path stays as the fallback for tiny `n`, `limit` (top-k), and the
near-sorted regime where radix is intrinsically weak (documented blind spot).

**Expected win (measured elsewhere, must re-measure on Arrow):** evolve radix
argsort vs comparison argsort was 8.9× (uint32, 1M random), 4.2–8.2× (uint64),
20× (low-cardinality), and even ordered input untouched. Arrow's `sort_primitive`
is the same comparison pattern, so a real-but-smaller win is plausible.

**Risk:** low. Additive fast-path behind a dtype+size+`limit.is_none()` gate;
fallback unchanged; stability matters (Arrow's index sort is stable for ties —
a stable LSD radix preserves that, an unstable one does not, so this must be
handled). Bench harness exists: `arrow/benches/sort_kernel.rs`.

### 2. Herf float radix — same PR, low marginal cost

Arrow **already** has the IEEE→ordered-int flip (`lib.rs:243`), so float argsort
can reuse the same radix kernel as item 1 after the flip. evolve measured
3.3–3.7× on random f64 — but **lower-cardinality floats lose** (the ordered-int
range is too wide for counting-collapse), so this needs the same cardinality
gate. Low novelty, decent impact, natural rider.

### 3. Deferred-materialization row sort — the novel bet, prototype only

This is the brainstorm's headline ("the one place the primitive could *advance*
Arrow's SOTA"). The idea: encode only a fixed-width prefix + row index, radix
that, and encode/compare deeper bytes **only** for prefix-collision runs —
avoiding the 2–3× full-row materialization for the common case where prefixes
already disambiguate.

**Honest assessment after reading the code:**
- It is **genuinely not present** in the row format — encoding is eager
  (confirmed above). So the *encoding-time* deferral is unexplored.
- BUT Arrow already captures most of the *comparison-time* benefit via the
  4-byte prefix short-circuit (`sort_bytes`, ByteView `cmp`). The remaining
  win is narrower than the brainstorm implies: it's the **allocation/encode**
  cost of deep bytes, not the comparison cost.
- Difficulty is **high**: `row_lengths`/`append`/`encode_column` touch all
  ~50 data types incl. nested struct/list/union/REE where "prefix" is
  ambiguous; you need a two-phase encoder + a tie-break re-encode path.
- Win is **narrow**: large multi-column string-heavy sorts with few prefix
  collisions. Regresses small/medium sorts (prefix overhead > savings).

**Recommendation:** scope a *standalone prototype* (a new sort kernel that
takes only the sort columns, encodes an 8–16B prefix + u32 index, radix-sorts,
and re-encodes collision runs) and benchmark against `RowConverter` + sort on
`arrow/benches/row_format.rs` / `lexsort.rs`. If it wins materially on a
realistic string-heavy multi-column case, *that* is the publishable result
(deferred materialization vs Arrow `RowConverter`, on ARM especially). If not,
we've cheaply retired the brainstorm's biggest open bet with evidence.

---

## Verification of brainstorm claims

| Brainstorm claim | Verdict | Evidence |
|---|---|---|
| "Arrow fully materializes the comparable key" | **TRUE** | `arrow-row/src/lib.rs:919–1019`, `1674` |
| "deferred materialization could advance Arrow's SOTA" | **plausible but narrowed** | comparison-time prefix already exists (`sort.rs:352`, `cmp.rs:740`); only encode-time deferral is open |
| "Arrow row format = normalized keys, impact not novelty" | **TRUE** for the encoding; the *radix-instead-of-comparison* sort over primitives is the un-taken, low-risk impact |
| Herf float encoding would be new to Arrow | **FALSE** | Arrow already flips IEEE floats (`lib.rs:243`) |

## Suggested next step

Implement item 1+2 (radix argsort for primitive + float columns) as the first,
low-risk PR with a real `sort_kernel.rs` benchmark, and separately spike item 3
as a throwaway prototype to settle the novelty question with numbers. I can
start either on request.

---
*Code citations are against this branch's checkout of arrow-rs. evolve numbers
are from the evolve repo's `sort/{NUMPY_BENCH,UNIVERSAL_SORT_LOG,KEYREADER_LOG,
BENCHMARKS_*}.md` and are cross-architecture measurements, not Arrow benchmarks.*
