# Execution authority

## Frozen method

- Local specification: `Known-g Physical Turbulence Bridge v0.1 实施方案.md`
- Main seed: `20260826`
- Frozen severity grid: `[0.75, 1.50, 2.25, 3.00, 3.75, 4.50]`
- Frozen realization:
  - `epsilon_zernike`: `[36, 512, 512]`, float32
  - `epsilon_tilt`: `[512, 512, 2]`, float32

The user's execution addendum dated 2026-08-26 overrides stale engineering shape
estimates in the Markdown but does not alter the method.

## Frozen upstream

- Repository: `https://github.com/VITA-Group/PiRN.git`
- Commit: `12b9ef6a32c75143c23402a7ab8f843624ca99e8`
- Simulator: `assets/simulator_new.py`
- Simulator SHA-256: `5c5023ed0cd8db1c759a3eca7dd047dd5169d9f761133ba0f29a4ceb8e20ddd4`
- P2S model SHA-256: `578c3cd09c5c44e66c1086ffc0aadcab99542b74d27b932e9ca601a3faf6a4d5`
- PSF dictionary SHA-256: `2204ace3e4b99f31d62d21d0ba19bae95c606393ba7df5111e2526172e6b24c6`
- I0/I2 table SHA-256: `aa99db54950df99465e5c8b7b7b16b7ed9236603b5c1a63d82dd8831b5712f2e`

The upstream files and assets are read-only. `physical_bridge.simulator` is a
device-compatible wrapper that must be checked against the frozen source math.

