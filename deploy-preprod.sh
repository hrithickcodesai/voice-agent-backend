#!/usr/bin/env bash
# Manual/local deploy of the preprod Cloudflare Worker + container.
# Run directly (./deploy-preprod) or via the pre-push hook (see .githooks/pre-push).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> Deploying preprod-voice-agent to Cloudflare"
npx wrangler deploy

# Hits the Worker-served UI rather than a container, so the check itself
# never spins one up.
echo "==> Smoke testing https://preprod-voice-agent.cwz.workers.dev/ui/"
curl -sf -o /dev/null -w "%{http_code}\n" --retry 5 --retry-delay 3 --retry-connrefused \
  https://preprod-voice-agent.cwz.workers.dev/ui/
echo "==> Deploy complete"
