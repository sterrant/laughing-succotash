#!/usr/bin/env python3
"""CP/M MBASIC Lunar Lander serial agent.

Designed for IMSAI serial console sessions where the game runs in MBASIC.

Features:
- Live serial mode (pyserial) with local echo of incoming console text.
- Configurable parser (prompt and state regex) to accommodate listing differences.
- Rule-based baseline controller.
- CSV turn logging for later model training.
- Replay mode for parser/controller testing using saved console logs.

Notes for adapting to other BASIC listings:
- Edit config.json parser.prompt_patterns to match the exact input prompt(s).
- Edit config.json parser.state_pattern if your listing prints columns differently.
- Some listings use CR only; configure serial.tx_line_ending accordingly.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json


@dataclass
class GameState:
    """Represents one parsed game row/turn state."""

    sec: Optional[float] = None
    altitude: Optional[float] = None
    velocity: Optional[float] = None
    fuel: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)
    raw_line: str = ""


class Parser:
    """Configurable text-stream parser for MBASIC Lunar Lander output."""

    def __init__(self, cfg: Dict):
        parser_cfg = cfg["parser"]
        automation_cfg = cfg.get("automation", {})

        self.prompt_patterns = [re.compile(p) for p in parser_cfg["prompt_patterns"]]
        self.state_pattern = re.compile(parser_cfg["state_pattern"])
        self.ignore_patterns = [re.compile(p) for p in parser_cfg.get("ignore_patterns", [])]

        # Optional additional key/value extractions per line.
        self.extra_patterns = {
            name: re.compile(pat)
            for name, pat in parser_cfg.get("extra_patterns", {}).items()
        }
        self.auto_responses = [
            (re.compile(item["pattern"]), str(item["response"]))
            for item in automation_cfg.get("auto_responses", [])
        ]

    def is_prompt(self, line: str) -> bool:
        return any(p.search(line) for p in self.prompt_patterns)

    def should_ignore(self, line: str) -> bool:
        return any(p.search(line) for p in self.ignore_patterns)

    def parse_state(self, line: str) -> Optional[GameState]:
        if self.should_ignore(line):
            return None

        m = self.state_pattern.search(line)
        if not m:
            return None

        groups = m.groupdict()
        state = GameState(
            sec=_to_float(groups.get("sec")),
            altitude=_to_float(groups.get("altitude")),
            velocity=_to_float(groups.get("velocity")),
            fuel=_to_float(groups.get("fuel")),
            raw_line=line.rstrip("\r\n"),
        )

        for name, pat in self.extra_patterns.items():
            em = pat.search(line)
            if em:
                try:
                    state.extras[name] = float(em.group(1))
                except (ValueError, IndexError):
                    pass

        return state

    def match_auto_response(self, line: str) -> Optional[str]:
        for pattern, response in self.auto_responses:
            if pattern.search(line):
                return response
        return None


class BasePolicy:
    """Policy interface so a learned policy can be plugged in later."""

    def choose_burn(self, state: Optional[GameState]) -> int:
        raise NotImplementedError


class RuleBasedPolicy(BasePolicy):
    """Simple baseline policy with conservative late-stage braking."""

    def __init__(self, cfg: Dict):
        self.max_burn = int(cfg["policy"].get("max_burn", 30))
        self.min_burn = int(cfg["policy"].get("min_burn", 0))

    def choose_burn(self, state: Optional[GameState]) -> int:
        if state is None:
            return 0

        alt = state.altitude if state.altitude is not None else 9999
        vel = state.velocity if state.velocity is not None else 0
        fuel = state.fuel if state.fuel is not None else 0

        if fuel <= 0:
            return 0

        # Piecewise heuristic tuned to the classic Ahl listing behavior.
        if alt > 700:
            target_v = 70
        elif alt > 400:
            target_v = 45
        elif alt > 200:
            target_v = 30
        elif alt > 80:
            target_v = 20
        else:
            target_v = 10

        error = vel - target_v
        if error <= 0:
            burn = 0
        elif error < 5:
            burn = 3
        elif error < 10:
            burn = 6
        elif error < 20:
            burn = 10
        else:
            burn = 15

        burn = max(self.min_burn, min(self.max_burn, burn))
        burn = min(burn, int(fuel))
        return burn


class TurnLogger:
    """CSV logger for each prompt/decision turn."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=[
                "timestamp",
                "mode",
                "sec",
                "altitude",
                "velocity",
                "fuel",
                "burn",
                "raw_line",
            ],
        )
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.fp.flush()

    def log_turn(self, mode: str, state: Optional[GameState], burn: int):
        self.writer.writerow(
            {
                "timestamp": time.time(),
                "mode": mode,
                "sec": _safe_num(state.sec if state else None),
                "altitude": _safe_num(state.altitude if state else None),
                "velocity": _safe_num(state.velocity if state else None),
                "fuel": _safe_num(state.fuel if state else None),
                "burn": burn,
                "raw_line": state.raw_line if state else "",
            }
        )
        self.fp.flush()

    def close(self):
        self.fp.close()


def run_live(cfg: Dict, parser: Parser, policy: BasePolicy, logger: TurnLogger) -> None:
    # Import here so replay mode can run without serial hardware dependencies installed.
    import serial

    serial_cfg = cfg["serial"]
    line_ending = _decode_line_ending(serial_cfg.get("tx_line_ending", "\\r"))

    ser = serial.Serial(
        port=serial_cfg["port"],
        baudrate=int(serial_cfg.get("baudrate", 19200)),
        bytesize=getattr(serial, serial_cfg.get("bytesize", "EIGHTBITS")),
        parity=getattr(serial, serial_cfg.get("parity", "PARITY_NONE")),
        stopbits=float(serial_cfg.get("stopbits", 1)),
        timeout=float(serial_cfg.get("timeout", 0.1)),
    )

    print(f"[agent] connected to {ser.port} @ {ser.baudrate} bps")
    automation_cfg = cfg.get("automation", {})
    startup_delay = float(automation_cfg.get("startup_delay_sec", 0))
    startup_commands = [str(c) for c in automation_cfg.get("startup_commands", [])]
    if startup_delay > 0:
        time.sleep(startup_delay)
    for cmd in startup_commands:
        payload = f"{cmd}{line_ending}"
        ser.write(payload.encode(serial_cfg.get("encoding", "ascii"), errors="replace"))
        print(f"[agent] startup_send={cmd!r}")

    buf = ""
    last_state: Optional[GameState] = None

    try:
        while True:
            chunk = ser.read(1024)
            if not chunk:
                continue

            text = chunk.decode(serial_cfg.get("encoding", "ascii"), errors="replace")

            # Echo incoming text exactly to local terminal.
            sys.stdout.write(text)
            sys.stdout.flush()

            buf += text
            lines = _split_complete_lines(buf)
            if lines:
                buf = lines.pop()  # last entry may be incomplete remainder

            for line in lines:
                state = parser.parse_state(line)
                if state:
                    last_state = state

                auto_reply = parser.match_auto_response(line)
                if auto_reply is not None:
                    payload = f"{auto_reply}{line_ending}"
                    ser.write(payload.encode(serial_cfg.get("encoding", "ascii"), errors="replace"))
                    print(f"[agent] auto_reply={auto_reply!r}")
                    continue

                if parser.is_prompt(line):
                    burn = policy.choose_burn(last_state)
                    payload = f"{burn}{line_ending}"
                    ser.write(payload.encode(serial_cfg.get("encoding", "ascii"), errors="replace"))
                    logger.log_turn("live", last_state, burn)
                    print(f"[agent] burn={burn}")

    except KeyboardInterrupt:
        print("\n[agent] interrupted, exiting.")
    finally:
        ser.close()


def run_replay(cfg: Dict, parser: Parser, policy: BasePolicy, logger: TurnLogger, replay_file: Path) -> None:
    """Replay a saved console text log for parser/policy testing."""

    if not replay_file.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_file}")

    last_state: Optional[GameState] = None
    text = replay_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    print(f"[replay] reading {replay_file}")
    for line in lines:
        # Echo to mimic live behavior.
        sys.stdout.write(line)

        state = parser.parse_state(line)
        if state:
            last_state = state

        auto_reply = parser.match_auto_response(line)
        if auto_reply is not None:
            print(f"[replay] auto_reply={auto_reply!r}")
            continue

        if parser.is_prompt(line):
            burn = policy.choose_burn(last_state)
            logger.log_turn("replay", last_state, burn)
            print(f"[replay] burn={burn}")

    sys.stdout.flush()
    print("[replay] done")


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_complete_lines(buf: str) -> List[str]:
    """Returns list including trailing remainder (possibly incomplete)."""
    out: List[str] = []
    start = 0
    for i, ch in enumerate(buf):
        if ch == "\n" or ch == "\r":
            out.append(buf[start : i + 1])
            start = i + 1
    out.append(buf[start:])
    return out


def _decode_line_ending(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _safe_num(v: Optional[float]) -> str:
    return "" if v is None else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="CP/M MBASIC Lunar Lander serial agent")
    ap.add_argument("--config", default="config.json", help="Path to JSON config")
    ap.add_argument(
        "--mode",
        choices=["live", "replay"],
        default="live",
        help="live: serial connected; replay: parse a saved log file",
    )
    ap.add_argument("--replay-file", help="Text log to replay in replay mode")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    parser = Parser(cfg)
    policy = RuleBasedPolicy(cfg)
    logger = TurnLogger(Path(cfg["logging"]["csv_path"]))

    try:
        if args.mode == "live":
            run_live(cfg, parser, policy, logger)
        else:
            if not args.replay_file:
                raise SystemExit("--replay-file is required in replay mode")
            run_replay(cfg, parser, policy, logger, Path(args.replay_file))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
