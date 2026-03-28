import json
import unittest
from pathlib import Path

from cpm_lander_agent import GameState, LookaheadRulePolicy, _simulate_step


class LookaheadPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))
        self.cfg['policy']['type'] = 'lookahead_rule'
        self.cfg['policy']['lookahead']['horizon_steps'] = 3

    def test_simulate_step_matches_expected_no_burn(self):
        # From listing behavior: with burn=0, acceleration is +5.
        alt_next, vel_next, fuel_next, td = _simulate_step(1000.0, 50.0, 150.0, 0)
        self.assertAlmostEqual(vel_next, 55.0)
        self.assertAlmostEqual(alt_next, 947.5)
        self.assertAlmostEqual(fuel_next, 150.0)
        self.assertIsNone(td)

    def test_lookahead_returns_reasonable_terminal_burn(self):
        p = LookaheadRulePolicy(self.cfg)
        high_v = GameState(sec=10, altitude=120, velocity=30, fuel=40)
        burn = p.choose_burn(high_v)
        self.assertGreaterEqual(burn, 1)
        self.assertLessEqual(burn, 15)


if __name__ == '__main__':
    unittest.main()
