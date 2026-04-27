# PR Summary: Add `COSINE_SIMILARITY` support in RediSearch

## Summary

This change adds public RediSearch support for `DISTANCE_METRIC COSINE_SIMILARITY` while preserving the existing internal cosine execution path.

The implementation is intentionally non-breaking:

- Existing `L2`, `IP`, and `COSINE` behavior remains unchanged.
- `COSINE_SIMILARITY` is exposed as a new public metric name.
- Internally, search/index execution continues to reuse cosine-distance behavior.
- No new VecSim ordering, heap, or comparator logic is introduced in RediSearch.

## Why

- Industry standard with vector databases is `cosine_similarity` with range [-1, 1].
- Many customers existing downstream apps assume cosine_similarity so lack of support adds friction for replacement.
- Many ecosystem integrations also assume this convention and require us to reverse engineer the number for support.
- Vector distance metric doesn't intuitively express exact opposite vectors like a negative number does.

## RediSearch module changes

### Metric parsing and metadata

RediSearch now accepts `COSINE_SIMILARITY` in schema creation and reports it back through metric stringification.

Touched areas:

- `src/spec.c`
- `src/vector_index.h`
- `src/vector_index.c`

### Internal cosine-path reuse

`COSINE_SIMILARITY` follows the same internal path as `COSINE` for query execution and vector normalization.

Touched areas:

- `src/vector_normalization.h`
- `src/iterators/hybrid_reader.c`

### Returned score semantics

For fields defined with `COSINE_SIMILARITY`, RediSearch converts exposed vector scores from cosine distance to cosine similarity at the output boundary:

- `similarity = 1 - distance`

This keeps internal ranking unchanged while presenting similarity-style results to users.

Touched areas:

- `src/vector_index.c`
- `src/iterators/hybrid_reader.c`

### Range query semantics

For `VECTOR_RANGE` on `COSINE_SIMILARITY` fields, RediSearch interprets the provided threshold as a similarity threshold and translates it before calling the existing range query path:

- internal radius = `1 - similarity_threshold`

The public input is validated against the similarity range `[-1, 1]`.

Touched area:

- `src/vector_index.c`

## Validation / tests

This PR adds focused RediSearch-side coverage for:

- `FT.CREATE` accepting `DISTANCE_METRIC COSINE_SIMILARITY`
- KNN result ordering matching cosine behavior
- returned scores being exposed as cosine similarity values
- range query thresholds being interpreted as similarity thresholds

## Design constraints preserved

- No changes to existing `COSINE`, `IP`, or `L2` semantics
- No new search-time cosine-similarity math in the RediSearch module
- No new ordering/comparator model
- No changes to the core cosine ranking behavior

## Notes

This PR is designed as a thin RediSearch-layer adaptation:

- keep cosine-based execution internally
- translate only the public metric name and exposed score/range semantics

If paired with the corresponding VectorSimilarity changes, this gives users a clean public `COSINE_SIMILARITY` metric without expanding the internal algorithm surface area.