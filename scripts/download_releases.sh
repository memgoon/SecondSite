#!/usr/bin/env bash
set -euo pipefail
python -m secondsite download --destination "${1:-artifacts}" --data-profile all --models-profile all
