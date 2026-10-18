#!/usr/bin/env bash
# Smoke-test a running CardioRisk API: readiness, a real prediction, input validation
# and metrics. Exits non-zero on the first failure (used by CI, Docker and Kubernetes).
#
# usage: scripts/smoke_test.sh [BASE_URL]           default http://127.0.0.1:8000
#        HOST_HEADER=cardiorisk.local scripts/smoke_test.sh http://<ingress-ip>
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
HDR=()
[[ -n "${HOST_HEADER:-}" ]] && HDR=(-H "Host: ${HOST_HEADER}")
SAMPLE="$(dirname "$0")/../sample_request.json"

echo "==> waiting for ${BASE}/ready"
for attempt in $(seq 1 30); do
  if curl -fsS "${HDR[@]}" "${BASE}/ready" >/dev/null 2>&1; then break; fi
  if [[ "${attempt}" == 30 ]]; then echo "FAIL: API not ready after 60 s"; exit 1; fi
  sleep 2
done

echo "==> GET /health";  curl -fsS "${HDR[@]}" "${BASE}/health"; echo
echo "==> GET /ready";   curl -fsS "${HDR[@]}" "${BASE}/ready"; echo

echo "==> POST /predict (sample_request.json)"
response="$(curl -fsS "${HDR[@]}" -X POST "${BASE}/predict" \
  -H "Content-Type: application/json" -H "X-Request-ID: smoke-test" -d @"${SAMPLE}")"
echo "${response}"
for field in prediction confidence probability_disease risk_band; do
  grep -q "\"${field}\"" <<<"${response}" || { echo "FAIL: '${field}' missing"; exit 1; }
done

echo "==> POST /predict with invalid input (expect 422)"
code="$(curl -s -o /dev/null -w '%{http_code}' "${HDR[@]}" -X POST "${BASE}/predict" \
  -H "Content-Type: application/json" -d '{"age": 5}')"
[[ "${code}" == "422" ]] || { echo "FAIL: expected 422, got ${code}"; exit 1; }
echo "got 422"

echo "==> GET /metrics (excerpt)"
curl -fsS "${HDR[@]}" "${BASE}/metrics" | grep -E '^cardiorisk_(predictions_total|model_loaded|model_info)'

echo "SMOKE TEST PASSED"
