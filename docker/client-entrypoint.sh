#!/usr/bin/env bash
# Test-DJ-Container: jackd (Dummy) -> tc netem -> jacktrip-Client -> client.py (Testsignal, Phasenanpassung).
set -euo pipefail

: "${DJ_NAME:?DJ_NAME fehlt}"
SERVER="${SERVER_HOST:-server}"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

jackd --no-realtime -d dummy -r 48000 -p 256 -C 0 -P 0 &
PIDS+=($!)
for _ in $(seq 1 50); do jack_lsp >/dev/null 2>&1 && break; sleep 0.3; done

if [ -n "${NETEM:-}" ]; then
  IFACE="$(ip -o -4 route show to default | awk '{print $5}' | head -n1)"
  # shellcheck disable=SC2086  # NETEM besteht aus mehreren Wörtern
  tc qdisc replace dev "${IFACE}" root netem ${NETEM}
  echo "netem auf ${IFACE}: ${NETEM}"
  tc qdisc show dev "${IFACE}"
else
  echo "netem: keine Störung (Profil clean)"
fi

# Server erreichbar? (TCP-Handshake-Port 4464)
for _ in $(seq 1 60); do
  (exec 3<>"/dev/tcp/${SERVER}/4464") 2>/dev/null && break
  sleep 1
done

jacktrip -C "${SERVER}" --remotename "${DJ_NAME}" -n 2 -q auto &
PIDS+=($!)

python /opt/itest/client.py &
PIDS+=($!)

code=0
wait -n || code=$?
echo "Prozess beendet (Exit $code)" >&2
exit "$code"
