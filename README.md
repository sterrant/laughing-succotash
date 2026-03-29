# CP/M Lunar Lander Agent (IMSAI Serial)

Python agent for MBASIC Lunar Lander running on an IMSAI over serial (19,200 bps, 8N1).

## Features
- Uses `pyserial` for live serial I/O.
- Echoes incoming console text to local terminal.
- Detects user-input prompts and sends burn values.
- Configurable parser for varying BASIC listings/prompts.
- Ignores the "plot of distance" graphics while parsing state.
- Rule-based baseline controller.
- CSV logging of each decision turn for later training.
- Replay mode for validating parser/controller against saved logs.
- Policy abstraction so rule-based policy can be swapped with a learned policy later.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
Edit `config.json`:
- `serial.port` for your USB/serial adapter.
- Default serial framing in this repo is set to **19,200 bps, 7E1** (`SEVENBITS`, `PARITY_EVEN`, `stopbits: 1`).
- `serial.tx_line_ending` must match your BASIC monitor; this repo now defaults to `"\\r\\n"` (CR+LF).
- `serial.tx_char_delay_sec` adds per-character pacing (default `0.02`) for older UARTs that can drop characters.
- `parser.prompt_patterns` / `parser.state_pattern` if your listing differs.
- `automation.startup_commands` to send startup input (for example `"RUN"`).
- `automation.auto_responses` for non-throttle prompts (for example `"Another mission" -> "Y"`).
- `parser.contact_pattern` + metric patterns parse terminal CONTACT outcome stats.
- `optimizer.enabled` turns on simple random-search parameter tuning between runs.

### Unattended startup / restart behavior
By default, the sample config is set to:
- send `RUN` once after connect (`automation.startup_commands`)
- respond `Y` to `Another mission` prompts (`automation.auto_responses`)

Disable either behavior by setting the list to `[]`.

### If `RUN` returns `Syntax error`
This is usually serial framing or line-ending mismatch. Verify:
- 19,200 bps, 7E1 (`SEVENBITS` + `PARITY_EVEN` + `stopbits: 1`)
- `tx_line_ending` is `\"\\r\\n\"` (CR+LF)
- If you still see truncated input like `RU`, increase `tx_char_delay_sec` (for example `0.05`).
- The agent also handles prompts printed without a newline (for example a bare `?`).

## Live mode
```bash
python3 cpm_lander_agent.py --mode live --config config.json
```

## Replay mode
Save a previous console session to a text file and run:
You can use the included `session.log` sample file for a quick test.

```bash
python3 cpm_lander_agent.py --mode replay --config config.json --replay-file session.log
```

Replay mode echoes the file, runs parser + policy, and appends decisions to CSV.

## Train a tiny neural policy (PyTorch)
After collecting turn data (`logs/turns.csv`), train a small MLP:

```bash
python3 train_policy.py --turn-csv logs/turns.csv --out-model models/policy.pt --out-norm models/policy_norm.json
```

This trains a simple behavior-cloning model from `(altitude, velocity, fuel, sec) -> burn`.

To switch the agent from rule policy to neural policy, change one config field:

```json
"policy": {
  "type": "neural"
}
```

The model/norm paths are configured via:
- `policy.neural_model_path`
- `policy.neural_norm_path`

Set `"policy.type": "rule"` to switch back.
Set `"policy.type": "lookahead_rule"` to use the short-horizon rule policy.

Lookahead tuning knobs:
- `policy.lookahead.burn_candidates_min`
- `policy.lookahead.burn_candidates_max`
- `policy.lookahead.burn_step`
- `policy.lookahead.terminal_burn_step`
- `policy.lookahead.terminal_altitude`
- `policy.lookahead.horizon_steps`
- `policy.lookahead.fuel_penalty_weight`

Fractional burns are supported (for example `4.5`), and terminal-phase search can use a finer step size.

## Analyze optimization runs from episodes.csv
Use the analyzer to summarize velocity stats, inspect best runs, and detect optimizer boundary clipping:

```bash
python3 analyze_episodes.py --episodes-csv logs/episodes.csv --config config.json --top-n 20 --recent 200
```

## Adapting to other BASIC listings
- **Prompt changes:** update `parser.prompt_patterns` regex list.
- **State line layout changes:** update `parser.state_pattern` named groups (`sec`, `altitude`, `velocity`, `fuel`).
- **Additional telemetry fields:** add regex entries under `parser.extra_patterns`.
- **Line ending quirks:** try `"\\r\\n"` or `"\\n"` if input is not accepted.

## Output logs
Per-turn decisions are appended to `logs/turns.csv` with:
- timestamp
- mode (`live`/`replay`)
- episode_id
- sec, altitude, velocity, fuel
- chosen burn
- raw parsed state line

Per-episode outcomes are appended to `logs/episodes.csv` with:
- episode_id, timestamp, mode
- contact_occurred
- touchdown_time_seconds
- landing_velocity_fps
- fuel_remaining_units
- reward (`-landing_velocity_fps`)
- run summary stats (`turns`, `min_altitude`, `max_speed`)
- `policy_params_json`
