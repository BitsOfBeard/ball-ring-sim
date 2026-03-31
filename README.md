# ball-ring-sim

Inspired by a Reddit video showcasing the idea, I built a faster and more scalable version in Python using NumPy, Numba, and Pygame.

If you want to waste some time watching balls collide, bounce, and multiply inside a ring, this is for you.

## What it is

`ball-ring-sim` is a 2D particle simulation where balls move inside a circular arena under gravity, collide with each other, and spawn new balls over time.

The simulation is optimized for high particle counts using preallocated NumPy arrays, Numba-compiled physics kernels, and a spatial grid for collision detection.

## Features

- Circular arena with gravity
- Ball-ball and ball-wall collisions
- Fast CPU simulation with Numba
- Spatial partitioning for collision speed-up
- Multiple HUD modes
- Optional rendering toggle for profiling
- Approximate runtime and performance statistics

## Physics note

This is a fast visual simulation, not a strict real-world physics model.

The current collision model is intentionally simplified. At high densities, the balls can behave more like a granular fluid or pseudo-liquid than like perfectly realistic marbles.

## Requirements

- Python 3.11 or newer recommended
- NumPy
- Numba
- Pygame

## Installation

Clone the repository:

```bash
git clone https://github.com/BitsOfBeard/ball-ring-sim.git
cd ball-ring-sim
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```
