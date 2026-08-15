# Archived: Candle+Metal Inference (Apple Silicon)

**Archived:** 2026-08-15

This directory contains the retired Candle+Metal inference implementation for Lumi Mamba-3 on Apple Silicon (M4 Pro). It is kept for reference only and is no longer maintained.

**Active inference implementation:** Native HIP for AMD Radeon AI PRO R9700 (gfx1201), located at [inference/](../../inference/) at the workspace root.

**Porting guide:** See [inference/METAL_TO_HIP.md](../../inference/METAL_TO_HIP.md) for the architectural decisions, kernel traps, and staged implementation plan for the HIP port.
