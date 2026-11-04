"""New phase-1 runtime; candidate/scientific primitives remain immutable."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT.parent/'v4_nafnet_sidd116m_candidate_20261005'
sys.path.insert(0,str(CANDIDATE))
