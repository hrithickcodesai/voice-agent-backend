#!/usr/bin/env bash
# Deploy of the prod Cloudflare Worker + container ("env.prod" in wrangler.jsonc).
# Run directly (./deploy-prod.sh) or via the pre-push hook when pushing the prod branch.
#
# One-time setup before the first prod deploy:
#   npx wrangler secret put ELEVENLABS_API_KEY   --env prod
#   npx wrangler secret put ELEVENLABS_VOICE_ID  --env prod
#   npx wrangler secret put OPENROUTER_API_KEY   --env prod
#   npx wrangler secret put PIPECAT_ICE_SERVERS  --env prod   # from a voice-agent-turn TURN key
#   openssl rand -base64 48 | tr -d '\n' | npx wrangler secret put SESSION_SECRET --env prod
# and add https://voice-agent.cwz.workers.dev to the Google OAuth client's
# authorized JavaScript origins.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REQUIRED_SECRETS=(ELEVENLABS_API_KEY ELEVENLABS_VOICE_ID OPENROUTER_API_KEY PIPECAT_ICE_SERVERS SESSION_SECRET)

# interactive runs confirm first; the pre-push hook / CI (no TTY) don't
if [[ -t 0 && "${SKIP_CONFIRM:-}" != "1" ]]; then
  read -r -p "Deploy to PROD (voice-agent)? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "Aborted."; exit 1; }
fi

# the Worker itself deploys fine without secrets, but every call would fail
echo "==> Checking prod secrets"
existing=$(npx wrangler secret list --env prod 2>/dev/null || echo "[]")
missing=()
for s in "${REQUIRED_SECRETS[@]}"; do
  grep -q "\"$s\"" <<<"$existing" || missing+=("$s")
done
if (( ${#missing[@]} )); then
  echo "Missing prod secrets: ${missing[*]}" >&2
  echo "Set them first - see the top of deploy-prod.sh." >&2
  exit 1
fi

# no TTY (stdin from /dev/null) = wrangler applies pending migrations without asking
echo "==> Applying D1 migrations (voice-agent-db)"
npx wrangler d1 migrations apply voice-agent-db --remote --env prod < /dev/null

echo "==> Deploying voice-agent to Cloudflare"
npx wrangler deploy --env prod

echo "==> Smoke testing https://voice-agent.cwz.workers.dev/ui/"
curl -sf -o /dev/null -w "%{http_code}\n" --retry 5 --retry-delay 3 --retry-connrefused \
  https://voice-agent.cwz.workers.dev/ui/
echo "==> Prod deploy complete"
