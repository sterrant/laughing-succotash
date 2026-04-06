import json
import unittest
from pathlib import Path

from cpm_lander_agent import GameState, PhysicsPolicy


class PhysicsPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))
        self.cfg['policy']['type'] = 'physics'

    def test_zero_fuel_returns_zero(self):
        p = PhysicsPolicy(self.cfg)
        s = GameState(sec=10, altitude=100, velocity=30, fuel=0)
        self.assertEqual(p.choose_burn(s), 0.0)

    def test_higher_velocity_requests_higher_burn(self):
        p = PhysicsPolicy(self.cfg)
        s1 = GameState(sec=10, altitude=100, velocity=10, fuel=100)
        s2 = GameState(sec=10, altitude=100, velocity=30, fuel=100)
        self.assertGreater(p.choose_burn(s2), p.choose_burn(s1))

    def test_output_is_clamped_to_fuel(self):
        p = PhysicsPolicy(self.cfg)
        s = GameState(sec=10, altitude=10, velocity=50, fuel=4.5)
        self.assertLessEqual(p.choose_burn(s), 4.5)

    def test_low_speed_can_command_sub_hover_burn(self):
        p = PhysicsPolicy(self.cfg)
        s = GameState(sec=10, altitude=1000, velocity=0, fuel=150)
        self.assertLess(p.choose_burn(s), 5.0)

    def test_initial_burn_delay_turns_holds_burn_at_zero(self):
        self.cfg['policy']['physics']['initial_burn_delay_turns'] = 7
        p = PhysicsPolicy(self.cfg)
        s_early = GameState(sec=3, altitude=1000, velocity=50, fuel=150)
        s_after = GameState(sec=7, altitude=700, velocity=20, fuel=120)
        self.assertEqual(p.choose_burn(s_early), 0.0)
        self.assertGreater(p.choose_burn(s_after), 0.0)


if __name__ == '__main__':
    unittest.main()
