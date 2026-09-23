#!/usr/bin/env bash
# Manual/local deploy of the preprod Cloudflare Worker + container.
# Run directly (./deploy-preprod) or via the pre-push hook (see .githooks/pre-push).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> Deploying preprod-voice-agent to Cloudflare"
npx wrangler deploy

echo "==> Smoke testing https://preprod-voice-agent.cwz.workers.dev/status"
curl -sf --retry 5 --retry-delay 3 --retry-connrefused \
  https://preprod-voice-agent.cwz.workers.dev/status
echo
echo "==> Deploy complete"
