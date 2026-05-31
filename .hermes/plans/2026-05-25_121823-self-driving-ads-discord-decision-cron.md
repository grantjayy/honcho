# Plan: Self-Driving Ads Discord Decision Cron

## Goal

Create infrastructure so a scheduled, non-agent job runs the existing `@self-driving-ads` campaign analysis on Grant's main Mac, posts the decision matrix to Discord channel `1508288184264556644`, and lets Grant select ads to turn off from Discord. When Grant confirms, the selected ads are paused via the existing Self Driving Ads API/Meta integration.

## Investigation summary

- Main Mac SSH target works via `ssh main-mac`.
- Project path on main Mac: `/Users/grantjordan/programs/0_projects/@self-driving-ads`.
- Repo uses `jj` for VCS; there are already unrelated working-copy changes, so implementation must avoid overwriting them.
- Existing run pattern from shell history:
  - `just analyze-campaign "$FB_ACCESS_TOKEN" 120253758332170386 200 "offsite_conversion.custom.1104636211541755"`
  - `just analyze-campaign "$FB_ACCESS_TOKEN" 120250184106790386 200 "offsite_conversion.custom.1104636211541755"`
- Existing server endpoints:
  - `POST /api/v1/analyze/loser`
  - `POST /api/v1/analyze/loser/campaign`
  - `POST /api/v1/analyze/pause`
- Existing shared models live in `packages/server_client/server_client/models.py`.
- Existing client wrapper lives in `packages/server_client/server_client/client.py`.
- Existing CLI prompts locally with `inquirer`, which is not usable for Discord decisions.
- `DISCORD_BOT_TOKEN`, `FB_ACCESS_TOKEN`, and `META_APP_SECRET` are present in the main Mac environment (values not inspected/exposed).
- Discord supports multi-select String Select components with `min_values` / `max_values`, max 25 options, and `custom_id` 1-100 chars. Components require an interaction-handling bot process; a webhook-only message cannot receive selections.

## Important architecture constraint

A Hermes `no_agent=true` cron script can run code and deliver stdout, but Hermes cron runs are not the right place to wait for Discord component interactions. The cron run should be short-lived. Discord select menu interactions require a live Discord bot/gateway listener (or an interactions webhook endpoint) after the report is posted.

Therefore the recommended architecture is two pieces:

1. **A short scheduled runner** invoked by Hermes cron over SSH that triggers analysis and posts/updates the Discord decision message.
2. **A persistent lightweight Discord decision bot on the main Mac** that receives select/confirm interactions and calls the existing pause endpoint.

This keeps the Hermes agent out of the loop while still allowing interactive Discord decisions.

## Proposed approach

### 1. Add reusable report/decision module to the project

Add a small operational package under `packages/cli/cli/discord_decisions.py` or a new script path like `scripts/self_driving_ads_discord.py`.

Responsibilities:

- Load configuration from env:
  - `DISCORD_BOT_TOKEN`
  - `SELF_DRIVING_ADS_DISCORD_CHANNEL_ID=1508288184264556644`
  - `FB_ACCESS_TOKEN`
  - `SELF_DRIVING_ADS_SERVER_URL=http://localhost:29171`
  - campaign configs for the two known campaigns:
    - `120253758332170386`
    - `120250184106790386`
    - `cpr_target=200`
    - `result_action_type=offsite_conversion.custom.1104636211541755`
- Call `ServerClient.analyze_loser_campaign(...)` for each configured campaign.
- Normalize results into a Discord-friendly decision matrix.
- Persist a run record locally (SQLite or JSON file under project state dir) containing:
  - `run_id`
  - message ID(s)
  - campaign IDs
  - ad IDs/names/status/reasons
  - selected ad IDs
  - decision state: pending/confirmed/applied/cancelled/failed

### 2. Use Discord embeds + multi-select components

Implement report output with Discord embeds, not Markdown tables.

Recommended Discord UX:

- Summary embed:
  - generated timestamp
  - campaign IDs
  - total/winner/healthy/loser counts
- Detail embed(s): one field per loser candidate:
  - ad name
  - campaign/adset/ad ID
  - spend/results/CPR
  - algorithm reason
- Multi-select menu:
  - label each option with truncated ad name + spend/CPR
  - value = ad ID or compact run/ad key
  - `max_values = min(25, number_of_candidates)`
- Buttons:
  - `Pause selected ads`
  - `Cancel / no changes`
  - optional `Refresh report`

If more than 25 loser candidates appear, split into multiple select menus/messages by campaign or page. Discord string selects allow max 25 options.

### 3. Add persistent interaction listener

Add a `listen` mode that runs a Discord bot process on the main Mac:

- Register persistent views/components with stable `custom_id` prefixes like:
  - `sda:select:<run_id>:<page>`
  - `sda:confirm:<run_id>`
  - `sda:cancel:<run_id>`
- On select interaction:
  - update the persisted selection
  - reply ephemerally with selected count/list
  - optionally edit the original message to show current selection
- On confirm interaction:
  - re-load selected ad IDs from persisted run state
  - call `POST /api/v1/analyze/pause` via `ServerClient.pause_ad(...)` for each selected ad
  - edit/append Discord message with success/failure results
  - mark the run `applied`
- Guardrails:
  - only allow Grant / authorized Discord user IDs to confirm if available
  - never auto-pause just because the analysis says loser
  - require explicit `Pause selected ads` button click after selecting

### 4. Add non-interactive scheduled runner mode

Add a `run-once` mode:

- Ensure the API server is reachable at `http://localhost:29171/health`.
- If not reachable, either:
  - fail with actionable Discord error, or
  - optionally start the server using the existing `just server` flow in a managed background process/launchd service (approval recommended before adding always-on server behavior).
- Analyze both campaigns.
- Post Discord decision report to channel `1508288184264556644`.
- Exit with code 0 after posting.

### 5. Add process management on the main Mac

For the persistent listener, use a macOS LaunchAgent on the main Mac rather than Hermes cron:

- `~/Library/LaunchAgents/com.moa.self-driving-ads.discord-decisions.plist`
- Runs from `/Users/grantjordan/programs/0_projects/@self-driving-ads`
- Command roughly:
  - `uv run python packages/cli/cli/discord_decisions.py listen`
- Logs to a project-local `logs/` directory.

This gives interaction handling across restarts and does not involve Hermes agent execution.

### 6. Add Hermes no-agent cron wrapper

Create a Hermes script under `~/.hermes/scripts/`, e.g. `self_driving_ads_decision_report.sh`, that only SSHes to the main Mac and runs the project script:

```bash
#!/usr/bin/env bash
set -euo pipefail
ssh main-mac 'cd /Users/grantjordan/programs/0_projects/@self-driving-ads && uv run python packages/cli/cli/discord_decisions.py run-once'
```

Create Hermes cron with:

- `no_agent=true`
- `script=self_driving_ads_decision_report.sh`
- `deliver=local` because the project script posts directly to Discord
- schedule: to be chosen/confirmed (e.g. daily morning or every N hours)

This makes Hermes only a scheduler/SSH trigger, not a reasoning agent.

## Files likely to change

In `/Users/grantjordan/programs/0_projects/@self-driving-ads`:

- `packages/cli/pyproject.toml`
  - add `discord.py>=2.4,<3` (or an equivalent Discord library)
- `packages/cli/cli/discord_decisions.py` (new)
  - runner + listener implementation
- `packages/server_client/server_client/client.py`
  - possibly add batch pause convenience wrapper, if useful
- `packages/server_client/server_client/models.py`
  - possibly add batch pause request/response models, if we decide to reduce HTTP calls
- `packages/server/features/analyze/losers/router.py`
  - optional batch pause endpoint if approved
- `packages/server/features/analyze/losers/pause_service.py`
  - optional batch pause service if approved
- `packages/cli/tests/test_discord_decisions.py` (new)
  - formatting/state/custom_id tests
- `justfile`
  - optional recipes: `discord-report`, `discord-listener`
- LaunchAgent plist on main Mac
  - likely outside repo unless we add a template under `ops/launchd/`
- Hermes wrapper on Hermes instance
  - `~/.hermes/scripts/self_driving_ads_decision_report.sh`

## Validation plan

1. Run unit tests:
   - `uv run pytest packages/cli/tests -q`
   - `uv run pytest packages/server/tests -q`
2. Start/verify server:
   - `just server`
   - `curl http://localhost:29171/health`
3. Dry-run report generation without pausing:
   - run `discord_decisions.py run-once --dry-run` and inspect generated payload/state.
4. Post a test report to Discord channel `1508288184264556644` with fake or dry-run data.
5. Verify interaction listener receives select menu + button interactions.
6. Confirm pause flow in dry-run mode first; then real mode only after explicit approval.
7. Verify Hermes cron is script-only:
   - `hermes cron list`
   - job has `no_agent: true`, `deliver: local`, and wrapper script name only.
8. Trigger cron manually once and confirm Discord message appears without Hermes agent response.

## Risks / tradeoffs

- **Discord interactions require a persistent bot/listener.** A cron script alone cannot reliably process a later multi-select decision.
- **Discord select menus cap at 25 options.** Need pagination/splitting for larger loser sets.
- **There are existing unrelated working-copy changes.** Implementation should preserve them and avoid broad formatting.
- **Access token freshness.** Current flow relies on `FB_ACCESS_TOKEN` being valid on main Mac. If token expiry is common, this may need to integrate with the separate DialerIO/token-refresh-style pattern later.
- **Server availability.** The decision bot can call the API only if the FastAPI server is running. We need to decide whether to manage the server as a LaunchAgent too, or require it to already be up.
- **Authorization.** Need Grant's Discord user ID or another explicit allowlist before enabling real pause actions.

## Open questions for approval

1. Schedule: what cadence/time should the report run? Daily? Hourly? Specific Austin time?
2. Should we include both historical campaign IDs (`120253758332170386` and `120250184106790386`) by default?
3. Do you want the persistent Discord listener installed as a LaunchAgent on the main Mac?
4. Should real pausing be guarded by a second confirm button after selection? I recommend yes.
5. Should I add a batch-pause endpoint to the server, or keep the existing one-ad-at-a-time pause endpoint for the first version?
