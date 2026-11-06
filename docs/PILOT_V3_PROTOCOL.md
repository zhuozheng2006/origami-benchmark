# Small three-model calibration, 2026-11-07

This is a calibration on existing development tasks, not a new held-out dataset or a general capability ranking.

- Tasks: O01, O02, O03 (accepted), and O04 (diagnostic only). Their diagrams, starting states, goals and reference actions are unchanged.
- One preregistered run per task and requested provider model: `gpt-6-astra`, `claude-opus-5-5`, `gemini-4-argon`. These are gateway declarations; returned names are recorded without claiming independent provider attestation.
- One public harness, 40 nonrefundable decisions per episode, 8192 requested output tokens per call, full public attempt history, original prompt and action interface. GPT uses its prior `reasoning_effort=high`; the other two retain their previous unspecified/default reasoning parameters. Differences are visible in model identity.
- New runs explicitly opt into `structural-v3` (`structure-and-diagram-motion-v3-candidate1`). Legacy reports are retained alongside the new grades. Final XYZ translation/yaw freedom does not relax physical, fold-memory, finish or ordered-motion requirements. O04 is not promoted into formal scores.
- HTTPS chat-completions, non-streaming, one HTTP attempt per model turn, 180-second response timeout. No hidden retry or endpoint fallback. API and transport errors remain in the denominator and are also reported separately from model failures.
- Physical simulation uses the same directional Codim-IPC backend on 4090B CPU groups 4, 5, 6. No GPU. Remote source hashes are checked before startup; native physical metadata must match the frozen reference signature before any paid call. Inference and the one shared ledger stay local.
- A reference O01 smoke run verifies the new transport before inference; it is installation evidence, not a model score. Whole-episode supervision is 5400 seconds; physical-command limits are 3600 seconds total and 600 seconds without a frame. Timeouts are environment errors, not physical impossibility.

## Cost control

The user increased the cumulative cap to CNY 150. The existing ledger is backed up and its ceiling explicitly migrated; all historical call rows and unresolved reservations remain unchanged. Runtime stopping threshold is CNY 140, with a separate shared CNY 30 allowance for this pilot. Gateway credit units use the user's conversion of 1 USD unit = CNY 1. No exchange conversion is applied.

Before this pilot: 1668 call records, CNY 48.3563545 estimated charges plus CNY 37.9532502 unresolved reservations = CNY 86.3096047 accounted occupancy. These estimates are not an invoice or actual payment total. Failed calls without complete usage retain their full reservation.

## Reproduction and comparison

`scripts/run_frozen_pilot.py freeze --config CONFIG.json --output NEW_STUDY --ledger SHARED.sqlite3` writes all 12 run UUIDs, model identities, source-manifest hash, prompt-policy hashes and the existing ledger baseline before execution. `execute --study STUDY --credentials PRIVATE_CONNECTIONS.json --ledger SHARED.sqlite3` uses those plans exactly once. It refuses a second execution marker, a substituted/changed ledger and existing run directories. Credentials never enter source freezes or observations.

`scripts/summarize_public.py` uses the preregistered per-model experiment plans to check coverage and identity. Diagnostic rows remain separate. `scripts/build_pilot_comparison.py` creates the local comparison page; it does not call models.

Historical main comparison selects the first `-01` run for every model/task cell. Five later O04 attempts are shown separately, with failures preserved, never as best-of replacement. Historical records lack the new identity hashes and used different transport retry rules; original scores, offline regrades and new real runs are distinct columns. Changes are descriptive across versions, not an isolated causal experiment on model ability or grading alone.
