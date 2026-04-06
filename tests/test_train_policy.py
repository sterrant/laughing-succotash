import csv
import tempfile
import unittest
from pathlib import Path

from train_policy import burn_to_class, class_to_burn, load_rows


class TrainPolicyLabelTests(unittest.TestCase):
    def test_burn_class_roundtrip_preserves_tenth_resolution(self):
        self.assertEqual(burn_to_class(4.9), 49)
        self.assertEqual(class_to_burn(49), 4.9)
        self.assertEqual(burn_to_class(4.95), 50)
        self.assertEqual(class_to_burn(50), 5.0)

    def test_load_rows_uses_fractional_bins_not_floor(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "turns.csv"
            with p.open("w", encoding="utf-8", newline="") as fp:
                w = csv.DictWriter(fp, fieldnames=["altitude", "velocity", "fuel", "sec", "burn"])
                w.writeheader()
                w.writerow({"altitude": "100", "velocity": "20", "fuel": "50", "sec": "3", "burn": "4.9"})
            rows = load_rows(p)
            self.assertEqual(len(rows), 1)
            _, y = rows[0]
            self.assertEqual(y, 49)


if __name__ == "__main__":
    unittest.main()
