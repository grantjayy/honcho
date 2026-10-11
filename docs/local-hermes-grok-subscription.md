# Local Hermes subscription route

The host runs `scripts/grok_subscription_bridge.py` through the separate LaunchAgent `com.hermes.honcho-grok-bridge`.
It listens on `127.0.0.1:8791`.
Colima containers reach that loopback service through `host.lima.internal`.
The bridge resolves refreshed `xai-oauth` credentials through the installed Hermes source for each request.
The bridge never reads the paid xAI API key and never changes providers on failure.

Apply these values to the local, ignored `config.toml`:

- `deriver.model_config.fallback.model`: `grok-4.3`
- `deriver.model_config.fallback.thinking_effort`: `medium`
- `deriver.model_config.fallback.overrides.base_url`: `http://host.lima.internal:8791/v1`
- `deriver.model_config.fallback.overrides.api_key_env`: `GROK_BRIDGE_TOKEN`
- `dialectic.levels.medium.model_config.model`: `grok-4.3`
- `dialectic.levels.medium.model_config.thinking_effort`: `medium`
- `dialectic.levels.medium.model_config.overrides.base_url`: `http://host.lima.internal:8791/v1`
- `dialectic.levels.medium.model_config.overrides.api_key_env`: `GROK_BRIDGE_TOKEN`

Keep both transports `openai`.
Keep every non-Grok route and the disabled summary unchanged.
Store only the independent bridge credential in `/Users/grantjordan/.hermes/state/grok-subscription-bridge/honcho.env`, with mode 0600.
Do not copy the subscription bearer into that file.

Start or recreate API and deriver with both Compose files:
`docker compose -f docker-compose.yml -f docker-compose.grok.yml up -d --no-deps --no-build api deriver`.
The second file adds the bridge credential file to those two services.
A start through only the first file removes this extra environment and makes Grok requests fail closed.
Database and Redis are outside this restart scope.

Health: `http://127.0.0.1:8791/health` and `http://127.0.0.1:8000/health`.
Synthetic real-backend canaries in both containers returned `HONCHO_BACKEND_OK` on Grok 4.3 medium.
The deriver fallback also returned typed JSON `HONCHO_STRUCTURED_OK` and a `synthetic_ping` tool call with `HONCHO_TOOL_OK`.
These tests changed no production memory records.
