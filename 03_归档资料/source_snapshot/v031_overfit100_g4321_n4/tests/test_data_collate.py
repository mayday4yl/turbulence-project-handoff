"""Mixed-G DataLoader integration without padded or transposed path metadata."""
from copy import deepcopy
import unittest

import torch
from torch.utils.data import DataLoader

from v031.data import collate_starts


class DataCollateTests(unittest.TestCase):
    def samples(self):
        return [
            {"clean": torch.full((3, 2, 2), index + 0.125, dtype=torch.float32),
             "observation": torch.full((3, 2, 2), index + 0.375, dtype=torch.float32),
             "G": torch.tensor(g, dtype=torch.float32),
             "severity_path": tuple(range(g, 0, -1)),
             "start_id": f"t{index}_G{g}", "trajectory_id": f"t{index}"}
            for index, g in enumerate([4, 1, 3, 2])
        ]

    def test_mixed_g_loader_retains_variable_length_paths_and_identity(self):
        samples = self.samples()
        original = deepcopy(samples)
        # The ordinary torch DataLoader is exercised, not only the collator.
        loader = DataLoader(samples, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=collate_starts)
        batch = next(iter(loader))
        self.assertEqual(batch["clean"].shape, (4, 3, 2, 2))
        self.assertEqual(batch["observation"].shape, (4, 3, 2, 2))
        self.assertEqual(batch["G"].shape, (4,))
        self.assertEqual(batch["G"].tolist(), [4., 1., 3., 2.])
        self.assertEqual(batch["severity_path"], ((4, 3, 2, 1), (1,), (3, 2, 1), (2, 1)))
        self.assertEqual(batch["start_id"], ["t0_G4", "t1_G1", "t2_G3", "t3_G2"])
        self.assertEqual(batch["trajectory_id"], ["t0", "t1", "t2", "t3"])
        for index in range(4):
            for key in ("clean", "observation", "G"):
                self.assertEqual(batch[key].dtype, torch.float32)
                torch.testing.assert_close(batch[key][index], original[index][key], rtol=0, atol=0)
                torch.testing.assert_close(samples[index][key], original[index][key], rtol=0, atol=0)

    def test_partial_final_batch_and_invalid_metadata(self):
        samples = self.samples()
        batches = list(DataLoader(samples, batch_size=3, collate_fn=collate_starts))
        self.assertEqual([batch["G"].numel() for batch in batches], [3, 1])
        self.assertEqual(batches[-1]["severity_path"], ((2, 1),))
        with self.assertRaises(ValueError):
            collate_starts([])
        samples[0]["severity_path"] = (4, 3)
        with self.assertRaises(ValueError):
            collate_starts(samples)


if __name__ == "__main__":
    unittest.main()
