# Infinite Fractal Explorer

A dynamic Mandelbrot set explorer with infinite zoom, smooth color transitions, and interactive effects.

## Features

- Infinite zoom capability with adaptive resolution
- Dynamic edge detection and smooth tracking
- Rotating view with trail effects
- Psychedelic color cycling
- Smooth performance optimization
- Real-time FPS monitoring

## Requirements

- Python 3.8+
- Pygame
- NumPy
- Numba

## Installation

1. Clone the repository:
```bash
git clone https://github.com/robvanprod/fractal-explorer.git
cd fractal-explorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the fractal explorer:
```bash
python fractal_renderer_pygame.py
```

Controls:
- ESC: Exit the application
- The fractal automatically zooms and follows interesting patterns

## Performance

The renderer automatically adjusts quality based on your system's performance to maintain smooth animation. The current FPS and scale factor are displayed in the window title.
