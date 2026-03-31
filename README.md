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

## Run

```bash
python main.py
```

The first run may take a little longer because Numba compiles the hot paths on startup.

## Controls

- `H` cycles HUD modes
- `R` toggles rendering
- `Esc` quits

## HUD modes

The simulation includes four HUD states:

- Off
- Ball count only
- Basic measured statistics
- Full measured statistics plus rough estimates

## Configuration

The simulation is currently configured by editing constants near the top of `main.py`.

The most important settings are:

- `WIDTH`, `HEIGHT`, `CIRCLE_RADIUS`
- `BALL_RADIUS`
- `BALL_RESTITUTION`
- `WALL_RESTITUTION`
- `GRAVITY`
- `MIN_SPEED`
- `MAX_SPEED`
- `FPS`
- `SUBSTEPS`
- `MAX_BALLS`
- `SPAWN_COOLDOWN`

Some values are derived automatically from the ball size, such as `CELL_SIZE` and the grid dimensions used for broad-phase collision detection.

### Default configuration

```python
WIDTH, HEIGHT = 800, 800
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
CIRCLE_RADIUS = 350

BALL_RADIUS = 2.0
BALL_RESTITUTION = 0.96
WALL_RESTITUTION = 0.98
GRAVITY = 400.0
MIN_SPEED = 0.0
MAX_SPEED = 2500.0

FPS = 60
SUBSTEPS = 4

MAX_BALLS = 250000
SPAWN_COOLDOWN = 0.0000001
```

### Tuning notes

Lower `BALL_RADIUS` lets more balls fit inside the ring, but it also changes collision density and rendering cost.

Increasing `SUBSTEPS` usually improves stability, but reduces performance.

`FAST_POINT_RENDER` is enabled automatically for very small balls.

If you want the simulation to feel less energetic, lower `BALL_RESTITUTION` and `WALL_RESTITUTION`.

## Performance notes

Performance depends heavily on particle radius, particle count, substep count, and whether rendering is enabled.

For very small balls, rendering can become the bottleneck before the physics simulation does.

If you want to isolate simulation performance, press `R` to disable rendering temporarily.

## Repository contents

```text
main.py
requirements.txt
README.md
LICENSE
.gitignore
```

## License

MIT License. See `LICENSE`.
