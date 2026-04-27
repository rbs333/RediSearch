# Implementation Plan: Add `COSINE_SIMILARITY` as a Fourth Vector Metric

## Goal

Add `COSINE_SIMILARITY` as a fourth accepted vector metric without changing the existing indexing or query execution approach.

The implementation should be non-breaking:

- Keep existing `L2`, `IP`, and `COSINE` behavior unchanged.
- Reuse the current `COSINE` execution path internally.
- Avoid introducing new ordering/comparator behavior in VecSim.
- Expose `COSINE_SIMILARITY` as a new public metric name with similarity-style output semantics.

## Alignment Review Against Existing Implementation

The plan should follow the code paths that already define and route vector metrics today.

All paths below are repository-relative.

### Public metric definition and parsing

- `deps/VectorSimilarity/src/VecSim/vec_sim_common.h`
  - Defines `VecSimMetric`.
  - Hosts the shared `VecSimMetric_IsCosineFamily(metric)` helper used to keep
    cosine and cosine-similarity branches in lockstep.
- `src/vector_index.h`
  - Defines metric string constants such as `VECSIM_METRIC_IP`, `VECSIM_METRIC_L2`, `VECSIM_METRIC_COSINE`.
- `src/spec.c`
  - `parseVectorField_GetMetric()` parses `DISTANCE_METRIC` during `FT.CREATE`.
- `src/vector_index.c`
  - `VecSimMetric_ToString()` stringifies metric values.
- `deps/VectorSimilarity/src/VecSim/utils/vec_utils.cpp`
  - VecSim-side `VecSimMetric_ToString()` (separate from RediSearch's helper).
- `deps/VectorSimilarity/src/python_bindings/bindings.cpp`
  - Test-facing `VecSimMetric` Python enum binding.

These are the places that have to learn the new public metric name.

### Existing cosine execution path to reuse

- `deps/VectorSimilarity/src/VecSim/spaces/spaces.cpp`
  - Routes `VecSimMetric_Cosine` to the low-level function family used for cosine distance.
- `deps/VectorSimilarity/src/VecSim/spaces/computer/preprocessors.h`
  - `static_assert` in `QuantPreprocessor` enumerates supported metrics.
- `deps/VectorSimilarity/src/VecSim/index_factories/components/preprocessors_factory.h`
  - Applies cosine preprocessing/normalization and computes
    `processed_bytes_count` (adds the trailing norm slot for integral types).
- `deps/VectorSimilarity/src/VecSim/index_factories/svs_factory.cpp`
  - Routes `VecSimMetric_Cosine` through the existing SVS metric mapping.
- `deps/VectorSimilarity/src/VecSim/vec_sim.cpp`
  - `VecSimParams_GetQueryBlobSize()` — sizes the query blob, including the
    appended norm slot for INT8/UINT8 cosine. Easy to miss.
- `deps/VectorSimilarity/src/VecSim/utils/vec_utils.cpp`
  - `VecSimParams_GetStoredDataSize()` — sizes the stored vector,
    same INT8/UINT8 cosine special case.
- `src/iterators/hybrid_reader.c`
  - Normalizes hybrid query vectors when the metric is cosine.

These are the internal switch points that must treat `COSINE_SIMILARITY`
identically to `COSINE`. To keep them in lockstep, every check uses
`VecSimMetric_IsCosineFamily(metric)` rather than open-coded
`metric == VecSimMetric_Cosine || metric == VecSimMetric_CosineSimilarity`.

### Score/output handling points

- `src/vector_index.c`
  - `createMetricIteratorFromVectorQueryResults()` copies returned metric values into result iterators.
- `src/iterators/hybrid_reader.c`
  - `HR_ReadKnnUnsortedSingle()` converts cosine distance to similarity for the
    non-hybrid KNN path. The hybrid read path intentionally does *not* convert
    (see Phase 3.4).
- `src/pipeline/pipeline_construction.c`
  - Selects metric-specific score normalization for hybrid execution.
- `src/vector_normalization.h`
  - Hosts `VectorNorm_*` and the shared `VecSimCosineDistanceToSimilarity()` helper.
- `src/result_processor.c`
  - Applies normalized vector scores in the pipeline.

These are the places to translate internal cosine-distance values into
externally visible cosine-similarity values when needed.

## Execution Plan

## Phase 1: Add the Fourth Public Metric

### 1.1 Extend the metric enum

Update `deps/VectorSimilarity/src/VecSim/vec_sim_common.h` to add:

- `VecSimMetric_CosineSimilarity`
- `VecSimMetric_IsCosineFamily(metric)` static-inline helper used by every
  cosine-aware branch in this plan.

This preserves the current metric model and simply adds a fourth option.

### 1.2 Add the public metric string

Update `src/vector_index.h` to add:

- `VECSIM_METRIC_COSINE_SIMILARITY "COSINE_SIMILARITY"`

### 1.3 Parse and stringify the new metric

Update:

- `src/spec.c`
  - Extend `parseVectorField_GetMetric()`.
- `src/vector_index.c`
  - Extend `VecSimMetric_ToString()`.
- `deps/VectorSimilarity/src/VecSim/utils/vec_utils.cpp`
  - Extend the VecSim-side `VecSimMetric_ToString()`.
- `deps/VectorSimilarity/src/python_bindings/bindings.cpp`
  - Add the new value to the `VecSimMetric` Python enum binding so the VecSim
    flow tests (`flow_*.py`) can reference it.

This keeps schema creation and metadata output aligned with the existing pattern for the other three metrics.

## Phase 2: Reuse the Existing COSINE Internal Path

### 2.1 Route `COSINE_SIMILARITY` anywhere `COSINE` is already routed

Extend existing cosine branches so that `VecSimMetric_CosineSimilarity` is handled identically to `VecSimMetric_Cosine` in the internal execution path.

Primary files:

- `deps/VectorSimilarity/src/VecSim/spaces/spaces.cpp`
- `deps/VectorSimilarity/src/VecSim/spaces/computer/preprocessors.h`
  (the `static_assert` enumerating supported quant metrics)
- `deps/VectorSimilarity/src/VecSim/index_factories/components/preprocessors_factory.h`
- `deps/VectorSimilarity/src/VecSim/index_factories/svs_factory.cpp`
- `deps/VectorSimilarity/src/VecSim/vec_sim.cpp`
  (`VecSimParams_GetQueryBlobSize` -- INT8/UINT8 cosine query blob sizing)
- `deps/VectorSimilarity/src/VecSim/utils/vec_utils.cpp`
  (`VecSimParams_GetStoredDataSize` -- INT8/UINT8 cosine stored-vector sizing)
- `src/iterators/hybrid_reader.c`

All of the above must use `VecSimMetric_IsCosineFamily(metric)` to decide
whether a cosine-only code path applies, instead of open-coding
`metric == VecSimMetric_Cosine || metric == VecSimMetric_CosineSimilarity`.

This means:

- The same normalization/preprocessing rules are reused.
- The same low-level cosine-distance functions are reused.
- The same index construction and search logic are reused.
- No new comparator or heap behavior is introduced.

### 2.2 Do not add new low-level metric math for VecSim search

Do **not** add new raw-similarity search functions in:

- `spaces/IP/IP.cpp`
- `spaces/IP/IP.h`
- heap/priority queue logic
- HNSW/BF ordering logic

Those changes would introduce a new search behavior model. They are unnecessary because cosine-distance ordering and cosine-similarity ordering are equivalent for KNN once you apply `similarity = 1 - distance` at the API boundary.

## Phase 3: Translate External Semantics at the Boundary

## Core rule

Internally:

- execute `COSINE_SIMILARITY` exactly like `COSINE`
- store/search/rank using cosine distance

Externally:

- present scores as cosine similarity
- accept cosine-similarity style thresholds where applicable

### 3.1 KNN queries

No ordering change is needed.

Reason:

- minimizing `1 - cosine_similarity`
- is equivalent to maximizing `cosine_similarity`

So the internal ranking stays unchanged.

### 3.2 Returned metric values

For fields defined with `COSINE_SIMILARITY`, convert the returned metric from cosine distance to cosine similarity:

- `similarity = 1 - distance`

Primary touch points:

- `src/vector_index.c`
  - for regular vector query replies returned through `createMetricIteratorFromVectorQueryResults()`
- `src/iterators/hybrid_reader.c`
  - `HR_ReadKnnUnsortedSingle()` only. The hybrid read path
    (`HR_ReadHybridUnsortedSingle()`) intentionally leaves the value as cosine
    distance and lets the hybrid scoring pipeline (Phase 3.4) consume it.
- `src/vector_normalization.h`
  - Hosts the shared `VecSimCosineDistanceToSimilarity()` helper used by both
    callers above. Clamps to `[-1, 1]` to absorb floating-point drift.

This is the key non-breaking adaptation: keep VecSim unchanged, translate only the exposed score.

### 3.3 Range queries

For `COSINE_SIMILARITY`, treat the user-provided threshold as a similarity threshold and convert it before calling the existing VecSim range query path:

- internal cosine distance radius = `1 - similarity_threshold`

Validation should enforce the public similarity range:

- allowed input range: `[-1, 1]`

Primary touch point:

- `src/vector_index.c`
  - before `VecSimIndex_RangeQuery(...)`

This preserves the existing range-query engine and only adapts the API semantics.

### 3.4 Hybrid score normalization

Hybrid normalization should continue using the existing metric-aware normalization pipeline.

Relevant files:

- `src/pipeline/pipeline_construction.c`
- `src/vector_normalization.h`
- `src/result_processor.c`

Implementation rule:

- do not invent a separate hybrid flow
- either map `VecSimMetric_CosineSimilarity` to the same normalization behavior as cosine distance
- or add a dedicated enum branch that reuses the same underlying transformation logic

Because the internal upstream value remains cosine distance until presentation, this should follow the existing cosine normalization pattern rather than introduce a new search-time calculation model.

Note: for hybrid queries, returning a hybrid score is expected and acceptable. `COSINE_SIMILARITY` does not need to force hybrid output fields into raw cosine-similarity `[-1, 1]` semantics when the query is intentionally using the hybrid scoring pipeline.

## Phase 4: Validation and Test Updates

Use existing test locations and patterns rather than creating a new test structure.

### 4.1 VecSim unit coverage

Update or extend:

- `deps/VectorSimilarity/tests/unit/test_spaces.cpp`
  - verify `VecSimMetric_CosineSimilarity` resolves through the same function-selection path as cosine
- `deps/VectorSimilarity/tests/unit/test_components.cpp`
  - verify preprocessing/normalization path matches cosine behavior, including
    the INT8/UINT8 norm-appending case (covers both the stored-vector and
    query-blob sizing fixes in `vec_utils.cpp` / `vec_sim.cpp`)
- `deps/VectorSimilarity/tests/unit/test_common.cpp`
  - regression test for `VecSimParams_GetQueryBlobSize` and
    `VecSimParams_GetStoredDataSize` returning the same size for `COSINE` and
    `COSINE_SIMILARITY` across `FLOAT32`, `FLOAT64`, `BFLOAT16`, `FLOAT16`,
    `INT8`, `UINT8`.

### 4.2 RediSearch integration coverage

Update or extend:

- `tests/pytests/test_vecsim.py`
  - `FT.CREATE` accepts `DISTANCE_METRIC COSINE_SIMILARITY` for every supported
    `(algorithm, type)` pair: `{FLAT, HNSW, SVS-VAMANA} x {FLOAT32, FLOAT64,
    BFLOAT16, FLOAT16, INT8, UINT8}` (skip the SVS combinations the engine
    already excludes for cosine).
  - `FT.INFO` reports `COSINE_SIMILARITY` (round-trip through
    `VecSimMetric_ToString`).
  - KNN ordering for `COSINE_SIMILARITY` matches the inverse ordering of
    `COSINE` over the same dataset (i.e. the doc-id sequences are identical
    when results are sorted by score in opposite directions).
  - For each result, `cosine_similarity == 1 - cosine_distance` to within a
    small float tolerance.
  - Range-query threshold is interpreted as a similarity value: a query with
    `RADIUS s` over `COSINE_SIMILARITY` returns the same docs as the same
    query with `RADIUS (1 - s)` over `COSINE`.
  - Range-query input validation rejects similarity thresholds outside
    `[-1, 1]`.
- `tests/pytests/test_hybrid_vector_normalizer.py`
  - Hybrid normalization for `COSINE_SIMILARITY` produces the same final
    pipeline scores as `COSINE` (the hybrid path keeps the value as cosine
    distance internally; see Phase 3.4).
- `tests/flow/flow_*.py` (VecSim)
  - At least one flow exercising `VecSimMetric_CosineSimilarity` end-to-end
    using the new Python enum binding from Phase 1.3.

### 4.3 Backward-compatibility checks

Explicitly verify that:

- `COSINE` behavior remains unchanged (returned values are still cosine
  *distance*, not similarity).
- `IP` behavior remains unchanged.
- `L2` behavior remains unchanged.
- Existing query ordering and score semantics for prior metrics do not
  regress.
- RDB load/save of indexes created on prior versions still works: a
  `COSINE` index round-trips as `COSINE`, and a `COSINE_SIMILARITY` index
  written on this version round-trips as `COSINE_SIMILARITY`.

## Phase 5: Documentation

Document only the externally visible addition:

- `COSINE` returns cosine distance: `1 - cosine_similarity`
- `COSINE_SIMILARITY` returns cosine similarity directly: `cosine_similarity`
- both use the same internal cosine execution path

This should be documented as a new metric name and output convention, not as a new indexing/search algorithm.

## Latency Considerations

The intended implementation keeps latency risk low because it reuses the existing cosine indexing and search path.

- No new search algorithm is introduced.
- No heap/comparator/order changes are introduced in VecSim.
- KNN execution keeps the same internal cosine-distance ranking behavior.
- Range queries add only a constant-time threshold translation from similarity to distance.
- Non-hybrid result reporting may add a small per-result conversion from distance to similarity (`1 - distance`).

Potential impact should therefore be limited to lightweight boundary translation work on returned results, which is expected to be negligible relative to vector search itself. The main thing to validate is that any result rewriting happens only at the output boundary and does not add extra preprocessing or extra distance computations inside the hot search path.

## Non-Breaking Guardrails

The implementation should preserve these constraints:

1. No changes to existing metric semantics.
2. No new VecSim heap/comparator/order model.
3. No new low-level cosine-similarity search math for HNSW/BF.
4. `COSINE_SIMILARITY` should be implemented by aliasing the internal cosine path and translating public inputs/outputs.
5. `modules/vector-sets/` is out of scope unless product requirements explicitly say this new metric must also exist for `VSET.*` commands.
