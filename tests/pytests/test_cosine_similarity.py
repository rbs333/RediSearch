import numpy as np
import scipy.spatial
from RLTest import Env

from common import getConnectionByEnv
from includes import *


def _assert_results(env, expected, actual, delta=1e-6):
    env.assertEqual(expected[0], actual[0])
    for i in range(1, len(expected), 2):
        env.assertEqual(expected[i], actual[i])
        env.assertAlmostEqual(expected[i + 1][1], float(actual[i + 1][1]), delta)


def test_cosine_similarity_knn_and_range_scores():
    env = Env(moduleArgs='DEFAULT_DIALECT 2')
    conn = getConnectionByEnv(env)

    params = ['TYPE', 'FLOAT32', 'DIM', '2', 'DISTANCE_METRIC', 'COSINE_SIMILARITY']
    env.expect('FT.CREATE', 'idx', 'SCHEMA', 'v', 'VECTOR', 'FLAT', str(len(params)), *params).ok()

    docs = {
        'a': np.array([1.0, 0.0], dtype=np.float32),
        'b': np.array([0.8, 0.2], dtype=np.float32),
        'c': np.array([0.0, 1.0], dtype=np.float32),
    }
    for doc_id, vector in docs.items():
        conn.execute_command('HSET', doc_id, 'v', vector.tobytes())

    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    expected_knn = [
        3,
        'a', ['score', 1.0 - scipy.spatial.distance.cosine(docs['a'], query_vec)],
        'b', ['score', 1.0 - scipy.spatial.distance.cosine(docs['b'], query_vec)],
        'c', ['score', 1.0 - scipy.spatial.distance.cosine(docs['c'], query_vec)],
    ]

    actual_knn = env.expect(
        'FT.SEARCH', 'idx', '*=>[KNN 3 @v $blob]=>{$yield_distance_as: score}',
        'PARAMS', '2', 'blob', query_vec.tobytes(),
        'SORTBY', 'score', 'DESC', 'RETURN', '1', 'score'
    ).res
    _assert_results(env, expected_knn, actual_knn)

    expected_range = [
        2,
        'a', ['score', expected_knn[2][1]],
        'b', ['score', expected_knn[4][1]],
    ]
    actual_range = env.expect(
        'FT.SEARCH', 'idx', '@v:[VECTOR_RANGE 0.95 $blob]=>{$yield_distance_as: score}',
        'PARAMS', '2', 'blob', query_vec.tobytes(),
        'SORTBY', 'score', 'DESC', 'RETURN', '1', 'score'
    ).res
    _assert_results(env, expected_range, actual_range)

    conn.execute_command('FT.DROPINDEX', 'idx', 'DD')