# Fractal-explorer

Fractal-explorer is a Pygame/Numba Mandelbrot renderer that automatically zooms, tracks edge-heavy regions, and resets at a configured zoom threshold.

## What It Is

This repo contains a visual fractal explorer with two renderers: one NumPy-oriented file and one Pygame implementation. The Pygame renderer computes Mandelbrot iterations, maps colors, follows strong edges, adjusts zoom speed, and displays the animation in a window.

## Current Status

Verified from code:

- `fractal_renderer_pygame.py` uses Pygame and NumPy.
- `compute_fractal_numba` is decorated for Numba acceleration.
- The renderer tracks edge points and transitions when zoom exceeds `1e3`.

`python -m pytest -q` reported `no tests ran` on 2026-05-28.

## Tech Stack

- Python
- Pygame
- NumPy
- Numba

## Limitations

This is a graphics experiment with no automated tests. Performance and stability depend on local graphics/runtime behavior and should be described only after running the app on the target machine.
