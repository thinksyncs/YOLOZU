#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p reports

python3 tools/yolozu.py doctor --output reports/doctor.json
python3 tools/report_dependency_licenses.py --output reports/dependency_licenses.json

echo "Wrote: reports/doctor.json"
echo "Wrote: reports/dependency_licenses.json"

