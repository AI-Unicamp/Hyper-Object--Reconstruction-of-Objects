# RainbowAI-2026-ICASSP-SPGC-GC7-Track2
We present a reproducible experimental framework for the ICASSP 2026 Hyperobject Challenge (Track 2), focusing on baseline engineering rather than architectural changes. Using the official organizer-provided baseline as a fixed reference, we standardize data handling, training and evaluation control, and experiment configuration to enable reliable comparison and rapid ablation. Our framework includes reproducible workflows, category-aware sampling, and practical I/O optimizations. Without modifying the baseline architecture, we achieve competitive performance (0.57146/SSC on the test-private split).

Hyperobject Challenge official repository: https://github.com/hyper-object/2026-ICASSP-SPGC
Hyperobject Challenge official page: https://hyper-object.github.io/

This repository contains:
- Complete reproducible tutorial (README)
- Config .yaml files
- CLI flags implemented
- (conferir) Conversion script .h5 -> .zarr
- (conferir) Code for SMOTE generation
