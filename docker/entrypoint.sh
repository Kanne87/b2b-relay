#!/usr/bin/env bash
# Startet im Container: jackd (Dummy-Treiber) -> JackTrip-Hub-Server (ohne Auto-Patching) -> b2b-relay.
# Prozesse werden über gespeicherte PIDs beendet; stirbt einer, endet der Container (Neustart durch Compose).
set -euo pipefail

JACK_RATE="${JACK_RATE:-48000}"
JACK_PERIOD="${JACK_PERIOD:-256}"
JACK_REALTIME="${JACK_REALTIME:-0}"            # 1 = jackd -R (braucht ulimit rtprio/memlock)
HUBPATCH="${JACKTRIP_HUBPATCH:-5}"             # 5 = kein Auto-Patching: das Routing macht b2b-relay
JACKTRIP_QUEUE="${JACKTRIP_QUEUE:-auto}"
JACKTRIP_UDP_BASE="${JACKTRIP_UDP_BASE:-61002}"
export B2B_SAMPLE_RATE="${JACK_RATE}"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

rt_flag="--no-realtime"; [ "${JACK_REALTIME}" = "1" ] && rt_flag="-R"
jackd ${rt_flag} -d dummy -r "${JACK_RATE}" -p "${JACK_PERIOD}" -C 0 -P 0 &
PIDS+=($!)

for _ in $(seq 1 50); do
  jack_lsp >/dev/null 2>&1 && break
  sleep 0.3
done
jack_lsp >/dev/null 2>&1 || { echo "jackd startet nicht" >&2; exit 1; }
echo "jackd läuft: ${JACK_RATE} Hz, Periode ${JACK_PERIOD}"

# shellcheck disable=SC2086
jacktrip -S -p "${HUBPATCH}" -q "${JACKTRIP_QUEUE}" -U "${JACKTRIP_UDP_BASE}" ${JACKTRIP_EXTRA:-} &
PIDS+=($!)
sleep 1
kill -0 "${PIDS[1]}" 2>/dev/null || { echo "jacktrip (Hub-Server) startet nicht" >&2; exit 1; }

python -m b2b_relay.live &
PIDS+=($!)

# Beendet sich einer der drei, enden alle.
code=0
wait -n || code=$?
echo "ein Prozess ist beendet (Exit $code), stoppe den Rest" >&2
exit "$code"
