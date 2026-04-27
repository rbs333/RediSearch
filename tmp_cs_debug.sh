#!/bin/bash
set +e
pkill -9 -f redis-server 2>/dev/null
sleep 1
RS_GLOBAL_DTORS=1 redis-server --port 26399 \
  --loadmodule /Users/robert.shelton/Documents/RediSearch/bin/macos-aarch64-release/search-community/redisearch.so \
  --logfile /tmp/rs_test.log --dir /tmp --daemonize yes --enable-debug-command yes
sleep 2
echo "PING:"
redis-cli -p 26399 ping
echo "--- HNSW COSINE_SIMILARITY:"
redis-cli -p 26399 FT.CREATE idx_h SCHEMA v VECTOR HNSW 6 TYPE FLOAT32 DIM 4 DISTANCE_METRIC COSINE_SIMILARITY
echo "--- FLAT COSINE_SIMILARITY:"
redis-cli -p 26399 FT.CREATE idx_f SCHEMA v VECTOR FLAT 6 TYPE FLOAT32 DIM 4 DISTANCE_METRIC COSINE_SIMILARITY
echo "--- SVS COSINE_SIMILARITY:"
redis-cli -p 26399 FT.CREATE idx_s SCHEMA v VECTOR SVS-VAMANA 6 TYPE FLOAT32 DIM 4 DISTANCE_METRIC COSINE_SIMILARITY
echo "--- FLAT COSINE (sanity):"
redis-cli -p 26399 FT.CREATE idx_fc SCHEMA v VECTOR FLAT 6 TYPE FLOAT32 DIM 4 DISTANCE_METRIC COSINE
echo "--- DEBUG INFO HNSW:"
redis-cli -p 26399 _FT.DEBUG VECSIM_INFO idx_h v 2>&1 | head -40
echo "--- DEBUG INFO FLAT:"
redis-cli -p 26399 _FT.DEBUG VECSIM_INFO idx_f v 2>&1 | head -40
echo "--- DEBUG INFO FLAT/COSINE:"
redis-cli -p 26399 _FT.DEBUG VECSIM_INFO idx_fc v 2>&1 | head -40
echo "--- LOG TAIL:"
tail -30 /tmp/rs_test.log
echo "--- SHUTDOWN:"
redis-cli -p 26399 SHUTDOWN NOSAVE 2>&1
exit 0
