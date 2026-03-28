import json
import unittest
from pathlib import Path

from cpm_lander_agent import EpisodeSummary, OutcomeExtractor, Parser, compute_reward


class ContactOutcomeParsingTests(unittest.TestCase):
    def setUp(self):
        cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))
        self.parser = Parser(cfg)
        self.extractor = OutcomeExtractor(self.parser)

    def _parse_block(self, text: str) -> EpisodeSummary:
        summary = EpisodeSummary(episode_id=1, mode='replay', start_timestamp=0.0)
        for line in text.splitlines():
            self.extractor.process_line(summary, line)
        return summary

    def test_extract_contact_metrics_example_1(self):
        sample = """***** CONTACT *****
Touchdown at 22.0887 seconds.
Landing velocity= 16.8226 feet/sec.
 0 units of fuel remaining.
"""
        summary = self._parse_block(sample)
        self.assertTrue(summary.contact_occurred)
        self.assertAlmostEqual(summary.touchdown_time_seconds, 22.0887)
        self.assertAlmostEqual(summary.landing_velocity_fps, 16.8226)
        self.assertAlmostEqual(summary.fuel_remaining_units, 0.0)
        self.assertAlmostEqual(summary.reward, -16.8226)

    def test_extract_contact_metrics_examples_2_and_3(self):
        sample_2 = """***** CONTACT *****
Touchdown at 27.0711 seconds.
Landing velocity= 35.3553 feet/sec.
 0 units of fuel remaining.
"""
        sample_3 = """***** CONTACT *****
Touchdown at 26.3246 seconds.
Landing velocity= 31.6228 feet/sec.
 0 units of fuel remaining.
"""
        s2 = self._parse_block(sample_2)
        s3 = self._parse_block(sample_3)

        self.assertAlmostEqual(s2.touchdown_time_seconds, 27.0711)
        self.assertAlmostEqual(s2.landing_velocity_fps, 35.3553)
        self.assertAlmostEqual(s3.touchdown_time_seconds, 26.3246)
        self.assertAlmostEqual(s3.landing_velocity_fps, 31.6228)

        # Lower landing velocity is better; reward is negative velocity.
        self.assertGreater(s3.reward, s2.reward)
        self.assertEqual(compute_reward(35.3553), -35.3553)


if __name__ == '__main__':
    unittest.main()
