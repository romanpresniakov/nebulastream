#!/bin/bash
# Measures network traffic and processing-time latency of the default and the partitioned TEAM setup
# (nes-systests/benchmark/TEAMPartitioning.test) on the running remote four-station cluster.
# Usage (cluster running, in build-rel/remote or with REMOTE_DIR set): ./team-bench.sh <label> [repetitions] [out.csv]
# <label> names the cluster configuration, e.g. COMPILER-4. Repetition 0 is a warm-up run.
set -euo pipefail
LABEL=${1:?usage: team-bench.sh <label> [repetitions] [out.csv]}
REPETITIONS=${2:-30}
OUT=${3:-team-bench.csv}
cd "${REMOTE_DIR:-$(dirname "$0")}"

TEST=/tmp/nebulastream/nes-systests/benchmark/TEAMPartitioning.test
WORKDIR=/tmp/nebulastream/build-rel/remote/systest-team-bench
WORKERS="sink-node station-01 station-02 station-03 station-04"
TOOL=nicolaka/netshoot
# Query numbers in TEAMPartitioning.test: network query, latency query.
declare -A NETWORK_QUERY=([default]=1 [partitioned]=3)
declare -A LATENCY_QUERY=([default]=2 [partitioned]=4)

# Runs a command in a worker's network namespace, with the rights to set and read packet counters.
in_netns() {
    local worker=$1; shift
    docker run --rm --net "container:$(docker compose ps -q "$worker")" --cap-add NET_ADMIN "$TOOL" "$@"
}

# Counting-only rules (no target) for everything a worker sends on the data (9090) and control (8080) ports.
for w in $WORKERS; do
    in_netns "$w" sh -c '
        iptables -N NES 2>/dev/null || true
        iptables -C OUTPUT -j NES 2>/dev/null || iptables -I OUTPUT -j NES
        iptables -F NES
        for p in 9090 8080; do iptables -A NES -p tcp --dport $p; iptables -A NES -p tcp --sport $p; done'
done

# One line per worker: eth0 tx, eth0 rx, data-plane tx, control-plane tx (bytes, cumulative).
counters() {
    for w in $WORKERS; do
        in_netns "$w" sh -c '
            port() { iptables -L NES -v -n -x | awk -v p="pt:$1" "\$0 ~ p {s += \$2} END {print s + 0}"; }
            echo "$(cat /sys/class/net/eth0/statistics/tx_bytes) $(cat /sys/class/net/eth0/statistics/rx_bytes) $(port 9090) $(port 8080)"'
    done
}

run_query() {
    docker compose exec -T systest build-rel/nes-systests/systest/systest \
        -t "$TEST:$1" -g TEAMBenchmark --remote \
        --clusterConfig /tmp/nebulastream/nes-systests/configs/topologies/four-stations.yaml \
        --data /tmp/nebulastream/nes-systests/testdata \
        --workingDir "$WORKDIR" > /dev/null 2>&1 || true
}

# Station data tx, station control tx, sink data tx, sink control tx, sink eth0 rx (bytes) of one network query.
measure_network() {
    local before after
    before=$(counters)
    run_query "$1"
    after=$(counters)
    paste <(echo "$before") <(echo "$after") | awk -v names="$WORKERS" '
        BEGIN { split(names, n, " ") }
        { if (n[NR] == "sink-node") { sdata = $7 - $3; sctrl = $8 - $4; srx = $6 - $2 }
          else { data += $7 - $3; ctrl += $8 - $4 } }
        END { printf "%d,%d,%d,%d,%d", data, ctrl, sdata, sctrl, srx }'
}

# last_ingest_ts, station_done_ts (empty for default), sink_arrival_ts, output_ts of one latency query.
measure_latency() {
    local result="TEAMPartitioning_$1.csv"
    docker compose exec -T systest sh -c "find '$WORKDIR/results' -name '$result' -delete 2>/dev/null || true"
    run_query "$1"
    docker compose exec -T systest sh -c "cat \$(find '$WORKDIR/results' -name '$result') 2>/dev/null | sed -n 2p" |
        awk -F, '{ if (NF == 5) printf "%s,,%s,%s", $3, $4, $5; else if (NF == 6) printf "%s,%s,%s,%s", $3, $4, $5, $6 }'
}

[ -s "$OUT" ] || echo "label,setup,rep,data_tx_stations,control_tx_stations,data_tx_sink,control_tx_sink,eth0_rx_sink,last_ingest_ts,station_done_ts,sink_arrival_ts,output_ts,latency_ms" > "$OUT"
for rep in $(seq 0 "$REPETITIONS"); do
    for setup in default partitioned; do
        network=$(measure_network "${NETWORK_QUERY[$setup]}")
        latency=$(measure_latency "${LATENCY_QUERY[$setup]}")
        if [ -z "$latency" ]; then
            echo "rep $rep $setup: no latency result, see $WORKDIR" >&2
            latency=",,,"
        fi
        echo "$LABEL,$setup,$rep,$network,$latency" | awk -F, -v OFS=, '{ $13 = ($9 != "" && $12 != "") ? $12 - $9 : ""; print }' | tee -a "$OUT"
    done
done

# Latency per setup over the measured repetitions (without the warm-up run 0): min, median, max in ms.
for setup in default partitioned; do
    awk -F, -v label="$LABEL" -v setup="$setup" 'NR > 1 && $1 == label && $2 == setup && $3 > 0 && $13 != "" { print $13 }' "$OUT" |
        sort -n | awk -v setup="$setup" '{ v[NR] = $1 }
            END { if (NR) printf "%-12s n=%d  min %d ms  median %d ms  max %d ms\n", setup, NR, v[1], v[int((NR + 1) / 2)], v[NR] }'
done
