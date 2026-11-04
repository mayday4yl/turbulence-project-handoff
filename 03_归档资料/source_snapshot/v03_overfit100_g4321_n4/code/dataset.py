"""Independent FP32 reader. All data refs resolve against the MAIN dataset root."""
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class OverfitDataset:
    def __init__(self, root, start_manifest="start_manifest.jsonl"):
        self.root = Path(root).resolve()
        self.starts = rows(self.root / start_manifest)
        self.sources = {r["source_id"]: r for r in rows(self.root / "source_manifest.jsonl")}
        self.trajectories = {r["trajectory_id"]: r for r in rows(self.root / "trajectory_manifest.jsonl")}
        if len({r["start_id"] for r in self.starts}) != len(self.starts):
            raise ValueError("Duplicate start ID")

    def array(self, ref, shape):
        path = (self.root / ref).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Data reference escaped dataset root")
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.float32 or x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f"Invalid array {ref}: {x.shape}/{x.dtype}")
        return x

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        r = self.starts[index]
        return {**r, "C": self.array(r["clean_ref"], (3, 256, 256)),
                "Y": self.array(r["initial_state_ref"], (3, 256, 256))}

    def diagnostic_xi(self, trajectory_id):
        r = self.trajectories[trajectory_id]
        return {"epsilon_zernike": self.array(r["epsilon_zernike_ref"], (36, 512, 512)),
                "epsilon_tilt": self.array(r["epsilon_tilt_ref"], (512, 512, 2))}
