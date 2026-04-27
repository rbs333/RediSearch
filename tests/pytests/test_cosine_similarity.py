import unittest

import numpy as np
from common import (
    VECSIM_ALGOS,
    VECSIM_DATA_TYPES,
    create_np_array_typed,
    get_vecsim_debug_dict,
    getConnectionByEnv,
)
from includes import *
from RLTest import Env

# FLAT and HNSW support all six VecSim data types for cosine; SVS-VAMANA is
# limited to FLOAT16 / FLOAT32 (mirrors the gating in test_vecsim.py).
_FLAT_HNSW_TYPES = VECSIM_DATA_TYPES + ["INT8", "UINT8"]
_SVS_TYPES = ["FLOAT32", "FLOAT16"]


def _supported_types(algo):
    return _SVS_TYPES if algo == "SVS-VAMANA" else _FLAT_HNSW_TYPES


def _knn_vectors_for_type(data_type):
    np.random.seed(7)
    dim = 4
    n = 6
    docs = np.random.rand(n, dim)
    query = np.random.rand(dim)

    if data_type == "INT8":
        docs = np.rint((docs - 0.5) * 100).astype(np.int8)
        query = np.rint((query - 0.5) * 100).astype(np.int8)
    elif data_type == "UINT8":
        docs = np.rint(docs * 100 + 1).astype(np.uint8)
        query = np.rint(query * 100 + 1).astype(np.uint8)

    return {f"d{i}": docs[i] for i in range(n)}, query


def _ids(res):
    return [res[i] for i in range(1, len(res), 2)]


def _scores(res, field):
    out = {}
    for i in range(1, len(res), 2):
        kv = res[i + 1]
        out[res[i]] = float(kv[kv.index(field) + 1])
    return out


def test_cosine_similarity_create_matrix():
    """COSINE_SIMILARITY parses across every supported (algorithm, type) pair
    and round-trips through FT.INFO."""
    env = Env(moduleArgs="DEFAULT_DIALECT 2", enableDebugCommand=True)
    conn = getConnectionByEnv(env)
    dim = 4

    for algo in VECSIM_ALGOS:
        for data_type in _supported_types(algo):
            idx = f'idx_{algo.replace("-", "_")}_{data_type}'
            params = [
                "TYPE",
                data_type,
                "DIM",
                dim,
                "DISTANCE_METRIC",
                "COSINE_SIMILARITY",
            ]
            env.expect(
                "FT.CREATE",
                idx,
                "SCHEMA",
                "v",
                "VECTOR",
                algo,
                str(len(params)),
                *params,
            ).ok()
            info = get_vecsim_debug_dict(env, idx, "v")
            env.assertEqual(
                info["METRIC"], "COSINE_SIMILARITY", message=f"{algo}/{data_type}"
            )
            conn.execute_command("FT.DROPINDEX", idx, "DD")


def test_cosine_similarity_knn_matches_cosine():
    """KNN over COSINE_SIMILARITY returns the same ids as COSINE, and
    per-result `similarity + distance == 1`."""
    env = Env(moduleArgs="DEFAULT_DIALECT 2", enableDebugCommand=True)
    conn = getConnectionByEnv(env)
    dim = 4

    for algo in VECSIM_ALGOS:
        for data_type in _supported_types(algo):
            docs, query = _knn_vectors_for_type(data_type)
            n = len(docs)
            conn.execute_command("FLUSHALL")
            params = [
                "TYPE",
                data_type,
                "DIM",
                dim,
                "DISTANCE_METRIC",
                "COSINE_SIMILARITY",
            ]
            env.expect(
                "FT.CREATE",
                "sim",
                "SCHEMA",
                "v",
                "VECTOR",
                algo,
                str(len(params)),
                *params,
            ).ok()
            params[-1] = "COSINE"
            env.expect(
                "FT.CREATE",
                "dist",
                "SCHEMA",
                "v",
                "VECTOR",
                algo,
                str(len(params)),
                *params,
            ).ok()

            for k, v in docs.items():
                conn.execute_command(
                    "HSET", k, "v", create_np_array_typed(v, data_type).tobytes()
                )
            qbytes = create_np_array_typed(query, data_type).tobytes()

            sim = env.cmd(
                "FT.SEARCH",
                "sim",
                f"*=>[KNN {n} @v $b]=>{{$yield_distance_as: s}}",
                "PARAMS",
                "2",
                "b",
                qbytes,
                "SORTBY",
                "s",
                "DESC",
                "RETURN",
                "1",
                "s",
            )
            dist = env.cmd(
                "FT.SEARCH",
                "dist",
                f"*=>[KNN {n} @v $b]=>{{$yield_distance_as: s}}",
                "PARAMS",
                "2",
                "b",
                qbytes,
                "SORTBY",
                "s",
                "ASC",
                "RETURN",
                "1",
                "s",
            )

            msg = f"{algo}/{data_type}"
            env.assertEqual(sim[0], dist[0], message=msg)
            env.assertEqual(_ids(sim), _ids(dist), message=msg)

            sim_s = _scores(sim, "s")
            dist_s = _scores(dist, "s")
            delta = 5e-2 if data_type in ("INT8", "UINT8") else 1e-3
            for k in sim_s:
                env.assertAlmostEqual(
                    sim_s[k] + dist_s[k], 1.0, delta=delta, message=f"{msg}/{k}"
                )


def test_cosine_similarity_hybrid_matches_cosine_distance_field():
    """Hybrid KNN keeps cosine-family vector scores in distance space, so
    COSINE_SIMILARITY should match COSINE for the yielded vector score field."""
    env = Env(moduleArgs="DEFAULT_DIALECT 2", enableDebugCommand=True)
    conn = getConnectionByEnv(env)
    dim = 4

    for algo in ("FLAT", "HNSW"):
        for data_type in ("INT8", "UINT8"):
            docs, query = _knn_vectors_for_type(data_type)
            n = len(docs)
            conn.execute_command("FLUSHALL")

            for idx, metric in (("sim", "COSINE_SIMILARITY"), ("dist", "COSINE")):
                params = [
                    "TYPE",
                    data_type,
                    "DIM",
                    dim,
                    "DISTANCE_METRIC",
                    metric,
                ]
                env.expect(
                    "FT.CREATE",
                    idx,
                    "SCHEMA",
                    "t",
                    "TEXT",
                    "v",
                    "VECTOR",
                    algo,
                    str(len(params)),
                    *params,
                ).ok()

            for k, v in docs.items():
                conn.execute_command(
                    "HSET",
                    k,
                    "t",
                    "keep",
                    "v",
                    create_np_array_typed(v, data_type).tobytes(),
                )

            qbytes = create_np_array_typed(query, data_type).tobytes()
            query_str = f"@t:keep=>[KNN {n} @v $b]=>{{$yield_distance_as: s}}"
            sim = env.cmd(
                "FT.SEARCH",
                "sim",
                query_str,
                "PARAMS",
                "2",
                "b",
                qbytes,
                "SORTBY",
                "s",
                "ASC",
                "RETURN",
                "1",
                "s",
            )
            dist = env.cmd(
                "FT.SEARCH",
                "dist",
                query_str,
                "PARAMS",
                "2",
                "b",
                qbytes,
                "SORTBY",
                "s",
                "ASC",
                "RETURN",
                "1",
                "s",
            )

            msg = f"{algo}/{data_type}"
            env.assertEqual(sim[0], dist[0], message=msg)
            env.assertEqual(_ids(sim), _ids(dist), message=msg)

            sim_s = _scores(sim, "s")
            dist_s = _scores(dist, "s")
            for k in sim_s:
                env.assertAlmostEqual(
                    sim_s[k], dist_s[k], delta=5e-2, message=f"{msg}/{k}"
                )


def test_cosine_similarity_range_matches_cosine():
    """Range query with COSINE_SIMILARITY radius `s` returns the same docs as
    COSINE with radius `1 - s`."""
    env = Env(moduleArgs="DEFAULT_DIALECT 2", enableDebugCommand=True)
    conn = getConnectionByEnv(env)
    dim = 3
    np.random.seed(11)
    docs = {f"d{i}": np.random.rand(dim) for i in range(10)}
    query = np.random.rand(dim)

    for algo in ("FLAT", "HNSW"):
        conn.execute_command("FLUSHALL")
        env.expect(
            "FT.CREATE",
            "sim",
            "SCHEMA",
            "v",
            "VECTOR",
            algo,
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            dim,
            "DISTANCE_METRIC",
            "COSINE_SIMILARITY",
        ).ok()
        env.expect(
            "FT.CREATE",
            "dist",
            "SCHEMA",
            "v",
            "VECTOR",
            algo,
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            dim,
            "DISTANCE_METRIC",
            "COSINE",
        ).ok()
        for k, v in docs.items():
            conn.execute_command(
                "HSET", k, "v", np.array(v, dtype=np.float32).tobytes()
            )
        qbytes = np.array(query, dtype=np.float32).tobytes()

        for s in (0.99, 0.9, 0.5, 0.0, -0.5):
            sim = env.cmd(
                "FT.SEARCH",
                "sim",
                f"@v:[VECTOR_RANGE {s} $b]",
                "PARAMS",
                "2",
                "b",
                qbytes,
                "NOCONTENT",
            )
            dist = env.cmd(
                "FT.SEARCH",
                "dist",
                f"@v:[VECTOR_RANGE {1.0 - s} $b]",
                "PARAMS",
                "2",
                "b",
                qbytes,
                "NOCONTENT",
            )
            env.assertEqual(sim[0], dist[0], message=f"{algo}/s={s}")
            env.assertEqual(sorted(sim[1:]), sorted(dist[1:]), message=f"{algo}/s={s}")


def test_cosine_similarity_range_radius_validation():
    """Range queries with COSINE_SIMILARITY must use a radius in [-1, 1]."""
    env = Env(moduleArgs="DEFAULT_DIALECT 2", enableDebugCommand=True)
    env.expect(
        "FT.CREATE",
        "idx",
        "SCHEMA",
        "v",
        "VECTOR",
        "FLAT",
        "6",
        "TYPE",
        "FLOAT32",
        "DIM",
        "2",
        "DISTANCE_METRIC",
        "COSINE_SIMILARITY",
    ).ok()
    qbytes = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    for bad in (1.5, -1.5, 2.0, -2.0):
        env.expect(
            "FT.SEARCH",
            "idx",
            f"@v:[VECTOR_RANGE {bad} $b]",
            "PARAMS",
            "2",
            "b",
            qbytes,
        ).error().contains("COSINE_SIMILARITY must be in [-1, 1]")
    # Boundary values must be accepted.
    for ok in (1.0, -1.0, 0.0):
        env.cmd(
            "FT.SEARCH", "idx", f"@v:[VECTOR_RANGE {ok} $b]", "PARAMS", "2", "b", qbytes
        )
