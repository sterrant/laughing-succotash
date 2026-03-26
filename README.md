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

## Adapting to other BASIC listings
- **Prompt changes:** update `parser.prompt_patterns` regex list.
- **State line layout changes:** update `parser.state_pattern` named groups (`sec`, `altitude`, `velocity`, `fuel`).
- **Additional telemetry fields:** add regex entries under `parser.extra_patterns`.
- **Line ending quirks:** try `"\\r\\n"` or `"\\n"` if input is not accepted.

## Output logs
Decisions are appended to `logs/turns.csv` with:
- timestamp
- mode (`live`/`replay`)
- sec, altitude, velocity, fuel
- chosen burn
- raw parsed state line
