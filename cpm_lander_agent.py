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
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
        self.contact_pattern = re.compile(
            parser_cfg.get("contact_pattern", r"^\*{5}\s*CONTACT\s*\*{5}\s*$")
        )
        self.touchdown_pattern = re.compile(
            parser_cfg.get(
                "touchdown_pattern",
                r"Touchdown at\s*(?P<touchdown_time_seconds>[-+]?\d+(?:\.\d+)?)\s*seconds\.",
            )
        )
        self.landing_velocity_pattern = re.compile(
            parser_cfg.get(
                "landing_velocity_pattern",
                r"Landing velocity=\s*(?P<landing_velocity_fps>[-+]?\d+(?:\.\d+)?)\s*feet/sec\.",
            )
        )
        self.fuel_remaining_pattern = re.compile(
            parser_cfg.get(
                "fuel_remaining_pattern",
                r"(?P<fuel_remaining_units>[-+]?\d+(?:\.\d+)?)\s*units of fuel remaining\.",
            )
        )

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

    def is_contact(self, line: str) -> bool:
        return bool(self.contact_pattern.search(line.strip()))

    def parse_touchdown_time(self, line: str) -> Optional[float]:
        m = self.touchdown_pattern.search(line)
        if not m:
            return None
        return _to_float(m.group("touchdown_time_seconds"))

    def parse_landing_velocity(self, line: str) -> Optional[float]:
        m = self.landing_velocity_pattern.search(line)
        if not m:
            return None
        return _to_float(m.group("landing_velocity_fps"))

    def parse_fuel_remaining(self, line: str) -> Optional[float]:
        m = self.fuel_remaining_pattern.search(line)
        if not m:
            return None
        return _to_float(m.group("fuel_remaining_units"))


@dataclass
class EpisodeSummary:
    episode_id: int
    mode: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    contact_occurred: bool = False
    touchdown_time_seconds: Optional[float] = None
    landing_velocity_fps: Optional[float] = None
    fuel_remaining_units: Optional[float] = None
    reward: Optional[float] = None
    turns: int = 0
    min_altitude: Optional[float] = None
    max_speed: Optional[float] = None
    policy_params_json: str = "{}"

    def is_complete(self) -> bool:
        return (
            self.contact_occurred
            and self.touchdown_time_seconds is not None
            and self.landing_velocity_fps is not None
            and self.fuel_remaining_units is not None
        )


class OutcomeExtractor:
    """Extract terminal CONTACT metrics and compute reward."""

    def __init__(self, parser: Parser):
        self.parser = parser

    def process_line(self, summary: EpisodeSummary, line: str) -> None:
        if self.parser.is_contact(line):
            summary.contact_occurred = True
        td = self.parser.parse_touchdown_time(line)
        if td is not None:
            summary.touchdown_time_seconds = td
        lv = self.parser.parse_landing_velocity(line)
        if lv is not None:
            summary.landing_velocity_fps = lv
            summary.reward = compute_reward(lv)
        fr = self.parser.parse_fuel_remaining(line)
        if fr is not None:
            summary.fuel_remaining_units = fr


class BasePolicy:
    """Policy interface so a learned policy can be plugged in later."""

    def choose_burn(self, state: Optional[GameState]) -> float:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return {}

    def set_params(self, params: Dict) -> None:
        del params


class RuleBasedPolicy(BasePolicy):
    """Simple baseline policy with conservative late-stage braking."""

    def __init__(self, cfg: Dict):
        policy_cfg = cfg["policy"]
        self.max_burn = int(policy_cfg.get("max_burn", 30))
        self.min_burn = int(policy_cfg.get("min_burn", 0))
        p = policy_cfg.get("parameters", {})
        self.params = {
            "target_v_high": float(p.get("target_v_high", 70)),
            "target_v_mid_high": float(p.get("target_v_mid_high", 45)),
            "target_v_mid": float(p.get("target_v_mid", 30)),
            "target_v_low": float(p.get("target_v_low", 20)),
            "target_v_final": float(p.get("target_v_final", 10)),
            "burn_small": float(p.get("burn_small", 3)),
            "burn_medium": float(p.get("burn_medium", 6)),
            "burn_large": float(p.get("burn_large", 10)),
            "burn_max": float(p.get("burn_max", 15)),
        }

    def get_params(self) -> Dict:
        return dict(self.params)

    def set_params(self, params: Dict) -> None:
        for k in self.params:
            if k in params:
                self.params[k] = float(params[k])

    def choose_burn(self, state: Optional[GameState]) -> float:
        if state is None:
            return 0

        alt = state.altitude if state.altitude is not None else 9999
        vel = state.velocity if state.velocity is not None else 0
        fuel = state.fuel if state.fuel is not None else 0

        if fuel <= 0:
            return 0

        # Piecewise heuristic tuned to the classic Ahl listing behavior.
        if alt > 700:
            target_v = self.params["target_v_high"]
        elif alt > 400:
            target_v = self.params["target_v_mid_high"]
        elif alt > 200:
            target_v = self.params["target_v_mid"]
        elif alt > 80:
            target_v = self.params["target_v_low"]
        else:
            target_v = self.params["target_v_final"]

        error = vel - target_v
        if error <= 0:
            burn = 0
        elif error < 5:
            burn = self.params["burn_small"]
        elif error < 10:
            burn = self.params["burn_medium"]
        elif error < 20:
            burn = self.params["burn_large"]
        else:
            burn = self.params["burn_max"]

        burn = max(float(self.min_burn), min(float(self.max_burn), float(burn)))
        burn = min(burn, float(fuel))
        return burn

    def _baseline_burn(self, altitude: float, velocity: float, fuel: float) -> float:
        if fuel <= 0:
            return 0
        s = GameState(sec=0.0, altitude=altitude, velocity=velocity, fuel=fuel)
        return RuleBasedPolicy.choose_burn(self, s)


class LookaheadRulePolicy(RuleBasedPolicy):
    """Rule policy with short-horizon candidate burn search."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        la = cfg["policy"].get("lookahead", {})
        self.burn_candidates_min = int(la.get("burn_candidates_min", 0))
        self.burn_candidates_max = int(la.get("burn_candidates_max", 15))
        self.horizon_steps = int(la.get("horizon_steps", 3))
        self.fuel_penalty_weight = float(la.get("fuel_penalty_weight", 0.05))
        self.burn_step = float(la.get("burn_step", 1.0))
        self.terminal_burn_step = float(la.get("terminal_burn_step", 0.5))
        self.terminal_altitude = float(la.get("terminal_altitude", 20.0))

    def choose_burn(self, state: Optional[GameState]) -> float:
        if state is None:
            return 0
        alt = state.altitude if state.altitude is not None else 9999.0
        vel = state.velocity if state.velocity is not None else 0.0
        fuel = state.fuel if state.fuel is not None else 0.0
        if fuel <= 0:
            return 0

        lo = max(float(self.min_burn), float(self.burn_candidates_min))
        hi = min(float(self.max_burn), float(self.burn_candidates_max), float(fuel))
        step = self.terminal_burn_step if alt <= self.terminal_altitude else self.burn_step
        candidates = _frange(lo, hi, step)

        best_burn = 0.0
        best_score = float("inf")
        for burn0 in candidates:
            score = self._score_candidate(alt, vel, fuel, float(burn0))
            if score < best_score:
                best_score = score
                best_burn = float(burn0)
        return best_burn

    def _score_candidate(self, alt: float, vel: float, fuel: float, burn0: float) -> float:
        a, v, f = alt, vel, fuel
        initial_fuel = fuel
        burn = float(min(max(0.0, burn0), f))
        touchdown_v = None
        req_burn = 5.0 + (max(0.0, vel) ** 2) / (2.0 * max(1.0, alt))
        req_burn = max(self.min_burn, min(self.max_burn, req_burn))
        initial_burn_penalty = 1.8 * abs(burn0 - req_burn)

        for step in range(max(1, self.horizon_steps)):
            if step > 0:
                    burn = self._baseline_burn(a, v, f)
            a_next, v_next, f_next, td_v = _simulate_step(a, v, f, burn)
            if td_v is not None:
                touchdown_v = td_v
                a, v, f = 0.0, td_v, f_next
                break
            a, v, f = a_next, v_next, f_next

        if touchdown_v is None:
            # Not yet at ground in horizon: estimate urgency from residual state.
            # Keep descending under control, avoid ascent/fuel dump behavior.
            target = self.params.get("target_v_final", 10.0)
            vel_term = abs(v - target)
            alt_term = 0.03 * max(0.0, a)
            ascent_penalty = 3.0 * max(0.0, -v)
            fuel_used_penalty = 0.06 * max(0.0, initial_fuel - f)
            dry_penalty = 60.0 if (f <= 0 and a > 40) else 0.0
            touchdown_like = vel_term + alt_term + ascent_penalty + fuel_used_penalty + dry_penalty
        else:
            touchdown_like = touchdown_v

        # Small penalty to avoid burning everything too early.
        fuel_penalty = self.fuel_penalty_weight * max(0.0, initial_fuel - f)
        best = touchdown_like + fuel_penalty + initial_burn_penalty
        return float(best)


class PhysicsPolicy(BasePolicy):
    """Physics-based controller using stopping-distance burn estimates."""

    def __init__(self, cfg: Dict):
        policy_cfg = cfg["policy"]
        self.min_burn = float(policy_cfg.get("min_burn", 0))
        self.max_burn = float(policy_cfg.get("max_burn", 30))
        phys = policy_cfg.get("physics", {})
        self.target_v_terminal = float(phys.get("target_v_terminal", 3.0))
        self.blend_altitude = float(phys.get("blend_altitude", 80.0))
        self.min_altitude_guard = float(phys.get("min_altitude_guard", 1.0))
        self.kp_velocity = float(phys.get("kp_velocity", 0.15))
        self.initial_burn_delay_turns = int(phys.get("initial_burn_delay_turns", 0))

    def choose_burn(self, state: Optional[GameState]) -> float:
        if state is None:
            return 0.0
        alt = state.altitude if state.altitude is not None else 9999.0
        vel = state.velocity if state.velocity is not None else 0.0
        fuel = state.fuel if state.fuel is not None else 0.0
        if fuel <= 0:
            return 0.0
        sec = state.sec if state.sec is not None else 0
        if sec < self.initial_burn_delay_turns:
            return 0.0

        h = max(self.min_altitude_guard, alt)
        v = max(0.0, vel)

        # Burn to cancel gravity + required decel to stop by touchdown.
        u_stop = 5.0 + (v * v) / (2.0 * h)

        # Near ground, blend with a terminal-speed-tracking burn.
        if alt <= self.blend_altitude:
            alpha = 1.0 - max(0.0, alt) / max(1e-6, self.blend_altitude)
            u_term = 5.0 + max(0.0, v - self.target_v_terminal)
            u = (1.0 - alpha) * u_stop + alpha * u_term
        else:
            u = u_stop

        # Small proportional velocity correction.
        u += self.kp_velocity * max(0.0, v - self.target_v_terminal)

        u = max(self.min_burn, min(self.max_burn, u))
        u = min(u, float(fuel))
        return float(u)

    def get_params(self) -> Dict:
        return {
            "policy_type": "physics",
            "target_v_terminal": self.target_v_terminal,
            "blend_altitude": self.blend_altitude,
            "min_altitude_guard": self.min_altitude_guard,
            "kp_velocity": self.kp_velocity,
            "initial_burn_delay_turns": self.initial_burn_delay_turns,
        }

    def set_params(self, params: Dict) -> None:
        for k in (
            "target_v_terminal",
            "blend_altitude",
            "min_altitude_guard",
            "kp_velocity",
            "initial_burn_delay_turns",
        ):
            if k in params:
                if k == "initial_burn_delay_turns":
                    setattr(self, k, max(0, int(params[k])))
                else:
                    setattr(self, k, float(params[k]))


class RandomSearchOptimizer:
    """Simple optimizer: keep best params by reward, sample around best."""

    def __init__(self, cfg: Dict):
        opt_cfg = cfg.get("optimizer", {})
        self.enabled = bool(opt_cfg.get("enabled", False))
        self.sigma = float(opt_cfg.get("sigma", 2.0))
        self.ranges = opt_cfg.get("param_ranges", {})
        self.best_reward = float("-inf")
        self.best_params: Optional[Dict] = None

    def maybe_update(self, policy: BasePolicy, summary: EpisodeSummary) -> None:
        if not self.enabled or summary.reward is None:
            return
        current = policy.get_params()
        if summary.reward > self.best_reward:
            self.best_reward = summary.reward
            self.best_params = dict(current)
            print(f"[optimizer] new best reward={summary.reward:.4f} params={self.best_params}")
        next_params = self._sample_around_best(current)
        policy.set_params(next_params)
        print(f"[optimizer] next params={policy.get_params()}")

    def _sample_around_best(self, fallback: Dict) -> Dict:
        base = self.best_params if self.best_params is not None else fallback
        out = dict(base)
        for name, r in self.ranges.items():
            lo = float(r.get("min"))
            hi = float(r.get("max"))
            center = float(base.get(name, (lo + hi) / 2))
            proposal = random.gauss(center, self.sigma)
            out[name] = max(lo, min(hi, proposal))
        return out


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
                "episode_id",
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

    def log_turn(self, mode: str, episode_id: int, state: Optional[GameState], burn: float):
        self.writer.writerow(
            {
                "timestamp": time.time(),
                "mode": mode,
                "episode_id": episode_id,
                "sec": _safe_num(state.sec if state else None),
                "altitude": _safe_num(state.altitude if state else None),
                "velocity": _safe_num(state.velocity if state else None),
                "fuel": _safe_num(state.fuel if state else None),
                "burn": _safe_num(burn),
                "raw_line": state.raw_line if state else "",
            }
        )
        self.fp.flush()

    def close(self):
        self.fp.close()


class EpisodeLogger:
    """CSV logger for per-episode outcomes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=[
                "episode_id",
                "timestamp",
                "mode",
                "contact_occurred",
                "touchdown_time_seconds",
                "landing_velocity_fps",
                "fuel_remaining_units",
                "reward",
                "turns",
                "min_altitude",
                "max_speed",
                "policy_params_json",
            ],
        )
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.fp.flush()

    def log_episode(self, summary: EpisodeSummary):
        self.writer.writerow(
            {
                "episode_id": summary.episode_id,
                "timestamp": summary.end_timestamp if summary.end_timestamp else time.time(),
                "mode": summary.mode,
                "contact_occurred": summary.contact_occurred,
                "touchdown_time_seconds": _safe_num(summary.touchdown_time_seconds),
                "landing_velocity_fps": _safe_num(summary.landing_velocity_fps),
                "fuel_remaining_units": _safe_num(summary.fuel_remaining_units),
                "reward": _safe_num(summary.reward),
                "turns": summary.turns,
                "min_altitude": _safe_num(summary.min_altitude),
                "max_speed": _safe_num(summary.max_speed),
                "policy_params_json": summary.policy_params_json,
            }
        )
        self.fp.flush()

    def close(self):
        self.fp.close()


def run_live(
    cfg: Dict,
    parser: Parser,
    policy: BasePolicy,
    turn_logger: TurnLogger,
    episode_logger: EpisodeLogger,
    optimizer: RandomSearchOptimizer,
) -> None:
    # Import here so replay mode can run without serial hardware dependencies installed.
    import serial

    serial_cfg = cfg["serial"]
    line_ending = _decode_line_ending(serial_cfg.get("tx_line_ending", "\\r"))
    tx_char_delay = float(serial_cfg.get("tx_char_delay_sec", 0.0))

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
        _send_line(
            ser,
            cmd,
            line_ending=line_ending,
            encoding=serial_cfg.get("encoding", "ascii"),
            tx_char_delay=tx_char_delay,
        )
        print(f"[agent] startup_send={cmd!r}")

    buf = ""
    last_state: Optional[GameState] = None
    outcome = OutcomeExtractor(parser)
    episode = _new_episode(1, "live", policy)

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
                    episode.min_altitude = (
                        state.altitude
                        if episode.min_altitude is None
                        else min(episode.min_altitude, state.altitude if state.altitude is not None else episode.min_altitude)
                    )
                    episode.max_speed = (
                        state.velocity
                        if episode.max_speed is None
                        else max(episode.max_speed, state.velocity if state.velocity is not None else episode.max_speed)
                    )
                outcome.process_line(episode, line)
                if episode.is_complete():
                    _finalize_episode(episode, episode_logger, optimizer, policy)
                    episode = _new_episode(episode.episode_id + 1, "live", policy)

                auto_reply = parser.match_auto_response(line)
                if auto_reply is not None:
                    _send_line(
                        ser,
                        auto_reply,
                        line_ending=line_ending,
                        encoding=serial_cfg.get("encoding", "ascii"),
                        tx_char_delay=tx_char_delay,
                    )
                    print(f"[agent] auto_reply={auto_reply!r}")
                    continue

                if parser.is_prompt(line):
                    burn = _quantize_burn(policy.choose_burn(last_state))
                    _send_line(
                        ser,
                        _format_burn(burn),
                        line_ending=line_ending,
                        encoding=serial_cfg.get("encoding", "ascii"),
                        tx_char_delay=tx_char_delay,
                    )
                    episode.turns += 1
                    turn_logger.log_turn("live", episode.episode_id, last_state, burn)
                    print(f"[agent] burn={burn}")

            # Some BASIC prompts (e.g. "?") are emitted without CR/LF.
            # Check remainder buffer so we can answer immediately.
            if buf:
                auto_reply = parser.match_auto_response(buf)
                if auto_reply is not None:
                    _send_line(
                        ser,
                        auto_reply,
                        line_ending=line_ending,
                        encoding=serial_cfg.get("encoding", "ascii"),
                        tx_char_delay=tx_char_delay,
                    )
                    print(f"[agent] auto_reply={auto_reply!r}")
                    buf = ""
                elif parser.is_prompt(buf):
                    burn = _quantize_burn(policy.choose_burn(last_state))
                    _send_line(
                        ser,
                        _format_burn(burn),
                        line_ending=line_ending,
                        encoding=serial_cfg.get("encoding", "ascii"),
                        tx_char_delay=tx_char_delay,
                    )
                    episode.turns += 1
                    turn_logger.log_turn("live", episode.episode_id, last_state, burn)
                    print(f"[agent] burn={burn}")
                    buf = ""

    except KeyboardInterrupt:
        print("\n[agent] interrupted, exiting.")
    finally:
        ser.close()


def run_replay(
    cfg: Dict,
    parser: Parser,
    policy: BasePolicy,
    turn_logger: TurnLogger,
    episode_logger: EpisodeLogger,
    optimizer: RandomSearchOptimizer,
    replay_file: Path,
) -> None:
    """Replay a saved console text log for parser/policy testing."""

    if not replay_file.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_file}")

    last_state: Optional[GameState] = None
    outcome = OutcomeExtractor(parser)
    episode = _new_episode(1, "replay", policy)
    text = replay_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    print(f"[replay] reading {replay_file}")
    for line in lines:
        # Echo to mimic live behavior.
        sys.stdout.write(line)

        state = parser.parse_state(line)
        if state:
            last_state = state
            episode.min_altitude = (
                state.altitude
                if episode.min_altitude is None
                else min(episode.min_altitude, state.altitude if state.altitude is not None else episode.min_altitude)
            )
            episode.max_speed = (
                state.velocity
                if episode.max_speed is None
                else max(episode.max_speed, state.velocity if state.velocity is not None else episode.max_speed)
            )
        outcome.process_line(episode, line)
        if episode.is_complete():
            _finalize_episode(episode, episode_logger, optimizer, policy)
            episode = _new_episode(episode.episode_id + 1, "replay", policy)

        auto_reply = parser.match_auto_response(line)
        if auto_reply is not None:
            print(f"[replay] auto_reply={auto_reply!r}")
            continue

        if parser.is_prompt(line):
            burn = _quantize_burn(policy.choose_burn(last_state))
            episode.turns += 1
            turn_logger.log_turn("replay", episode.episode_id, last_state, burn)
            print(f"[replay] burn={_format_burn(burn)}")

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


def _frange(lo: float, hi: float, step: float) -> List[float]:
    step = max(0.01, float(step))
    out: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        out.append(round(x, 4))
        x += step
    if not out:
        out = [round(lo, 4)]
    return out


def _simulate_step(altitude: float, velocity: float, fuel: float, burn: float):
    """One-second MBASIC-like update.

    Empirical from classic listing output:
    - acceleration term is roughly (5 - burn)
    - altitude update uses v + 0.5*a over 1 second
    """
    burn = float(max(0.0, min(float(fuel), burn)))
    a = 5.0 - burn
    alt_next = altitude - (velocity + 0.5 * a)
    vel_next = velocity + a
    fuel_next = max(0.0, fuel - burn)

    touchdown_velocity = None
    if alt_next <= 0.0:
        touchdown_velocity = _touchdown_velocity(altitude, velocity, a)
    return alt_next, vel_next, fuel_next, touchdown_velocity


def _touchdown_velocity(altitude: float, velocity: float, accel: float) -> float:
    """Velocity magnitude at touchdown within a one-second step."""
    if altitude <= 0:
        return abs(velocity)

    # Solve altitude - velocity*t - 0.5*accel*t^2 = 0 for t in [0, 1].
    aa = 0.5 * accel
    bb = velocity
    cc = -altitude

    if abs(aa) < 1e-9:
        t = altitude / max(1e-9, velocity)
    else:
        disc = bb * bb - 4 * aa * cc
        disc = max(0.0, disc)
        sqrt_disc = disc**0.5
        t1 = (-bb + sqrt_disc) / (2 * aa)
        t2 = (-bb - sqrt_disc) / (2 * aa)
        candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
        t = candidates[0] if candidates else 1.0
    v_touch = velocity + accel * t
    return abs(v_touch)


def _send_line(
    ser,
    text: str,
    *,
    line_ending: str,
    encoding: str,
    tx_char_delay: float,
) -> None:
    payload = f"{text}{line_ending}"
    if tx_char_delay <= 0:
        ser.write(payload.encode(encoding, errors="replace"))
        return

    # Some vintage systems overrun UART input at sustained host-side speed.
    # Send one byte at a time with a configurable delay.
    for ch in payload:
        ser.write(ch.encode(encoding, errors="replace"))
        time.sleep(tx_char_delay)


def _format_burn(burn: float) -> str:
    # Keep integers clean, but preserve fractional precision when needed.
    if abs(burn - round(burn)) < 1e-9:
        return str(int(round(burn)))
    return f"{burn:.1f}".rstrip("0").rstrip(".")


def _quantize_burn(burn: float) -> float:
    return round(float(burn), 1)


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _safe_num(v: Optional[float]) -> str:
    return "" if v is None else str(v)


def compute_reward(landing_velocity_fps: float) -> float:
    # Primary optimization target: lower landing velocity is better.
    return -float(landing_velocity_fps)


def _new_episode(episode_id: int, mode: str, policy: BasePolicy) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=episode_id,
        mode=mode,
        start_timestamp=time.time(),
        policy_params_json=json.dumps(policy.get_params(), sort_keys=True),
    )


def _finalize_episode(
    episode: EpisodeSummary,
    episode_logger: EpisodeLogger,
    optimizer: RandomSearchOptimizer,
    policy: BasePolicy,
) -> None:
    episode.end_timestamp = time.time()
    episode.policy_params_json = json.dumps(policy.get_params(), sort_keys=True)
    episode_logger.log_episode(episode)
    optimizer.maybe_update(policy, episode)
    print(
        "[episode] "
        f"id={episode.episode_id} "
        f"landing_velocity_fps={episode.landing_velocity_fps} "
        f"reward={episode.reward}"
    )


def build_policy(cfg: Dict) -> BasePolicy:
    policy_type = str(cfg.get("policy", {}).get("type", "rule")).lower()
    if policy_type == "neural":
        from neural_policy import NeuralPolicy

        return NeuralPolicy(cfg)  # type: ignore[return-value]
    if policy_type == "lookahead_rule":
        return LookaheadRulePolicy(cfg)
    if policy_type == "physics":
        return PhysicsPolicy(cfg)
    return RuleBasedPolicy(cfg)


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
    policy = build_policy(cfg)
    turn_logger = TurnLogger(Path(cfg["logging"]["csv_path"]))
    episode_logger = EpisodeLogger(Path(cfg["logging"]["episode_csv_path"]))
    optimizer = RandomSearchOptimizer(cfg)

    try:
        if args.mode == "live":
            run_live(cfg, parser, policy, turn_logger, episode_logger, optimizer)
        else:
            if not args.replay_file:
                raise SystemExit("--replay-file is required in replay mode")
            run_replay(
                cfg,
                parser,
                policy,
                turn_logger,
                episode_logger,
                optimizer,
                Path(args.replay_file),
            )
    finally:
        turn_logger.close()
        episode_logger.close()


if __name__ == "__main__":
    main()
