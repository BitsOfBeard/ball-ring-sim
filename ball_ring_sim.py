import math
import random
import sys
import time

import numpy as np
import pygame
from numba import njit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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

BG_COLOR = (12, 12, 12)
RING_COLOR = (52, 52, 52)
TEXT_COLOR = (220, 220, 220)

FAST_POINT_RENDER = BALL_RADIUS <= 1.5

CELL_SIZE = max(2, int(math.ceil(BALL_RADIUS * 2.0)))
GRID_W = (WIDTH + CELL_SIZE - 1) // CELL_SIZE
GRID_H = (HEIGHT + CELL_SIZE - 1) // CELL_SIZE
GRID_CELLS = GRID_W * GRID_H

HEX_PACKING_DENSITY = 0.9069
HUD_UPDATE_PERIOD = 0.5

HUD_OFF = 0
HUD_COUNT = 1
HUD_BASIC = 2
HUD_FULL = 3
HUD_MODE_COUNT = 4


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def integrate_and_wall(
    x,
    y,
    vx,
    vy,
    n,
    dt,
    gravity,
    cx,
    cy,
    arena_radius,
    ball_radius,
    wall_restitution,
):
    wall_hits = 0
    limit = arena_radius - ball_radius
    limit2 = limit * limit

    for i in range(n):
        vy[i] += gravity * dt
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt

        dx = x[i] - cx
        dy = y[i] - cy
        dist2 = dx * dx + dy * dy

        if dist2 > limit2:
            dist = math.sqrt(dist2)
            if dist > 0.0:
                nx = dx / dist
                ny = dy / dist

                dot = vx[i] * nx + vy[i] * ny
                if dot > 0.0:
                    vx[i] -= (1.0 + wall_restitution) * dot * nx
                    vy[i] -= (1.0 + wall_restitution) * dot * ny

                x[i] = cx + nx * (limit - 0.05)
                y[i] = cy + ny * (limit - 0.05)
                wall_hits += 1

    return wall_hits


@njit(cache=True)
def build_grid(
    x,
    y,
    n,
    cell_size,
    grid_w,
    grid_h,
    cell_id,
    cell_count,
    cell_start,
    cell_cursor,
    sorted_idx,
):
    for c in range(cell_count.shape[0]):
        cell_count[c] = 0

    for i in range(n):
        gx = int(x[i] / cell_size)
        gy = int(y[i] / cell_size)

        if gx < 0:
            gx = 0
        elif gx >= grid_w:
            gx = grid_w - 1

        if gy < 0:
            gy = 0
        elif gy >= grid_h:
            gy = grid_h - 1

        cid = gy * grid_w + gx
        cell_id[i] = cid
        cell_count[cid] += 1

    cell_start[0] = 0
    for c in range(cell_count.shape[0]):
        cell_start[c + 1] = cell_start[c] + cell_count[c]
        cell_cursor[c] = cell_start[c]

    for i in range(n):
        cid = cell_id[i]
        pos = cell_cursor[cid]
        sorted_idx[pos] = i
        cell_cursor[cid] += 1


@njit(cache=True, inline="always")
def resolve_pair(x, y, vx, vy, i, j, min_dist, min_dist2, restitution):
    eps = 1e-8

    dx = x[j] - x[i]
    dy = y[j] - y[i]
    dist2 = dx * dx + dy * dy

    if not math.isfinite(dist2):
        return 0, 0

    if dist2 >= min_dist2:
        return 0, 0

    if dist2 <= eps:
        rvx = vx[i] - vx[j]
        rvy = vy[i] - vy[j]
        rv2 = rvx * rvx + rvy * rvy

        if rv2 > eps:
            rv = math.sqrt(rv2)
            nx = rvx / rv
            ny = rvy / rv
        else:
            nx = 1.0
            ny = 0.0

        dist = min_dist
    else:
        dist = math.sqrt(dist2)
        if dist <= eps:
            return 0, 0
        nx = dx / dist
        ny = dy / dist

    rvx = vx[i] - vx[j]
    rvy = vy[i] - vy[j]
    dvn = rvx * nx + rvy * ny

    impulse_events = 0
    if dvn > 0.0:
        delta = 0.5 * (1.0 + restitution) * dvn
        vx[i] -= delta * nx
        vy[i] -= delta * ny
        vx[j] += delta * nx
        vy[j] += delta * ny
        impulse_events = 1

    overlap = min_dist - dist
    if overlap > 0.0:
        push = 0.5 * overlap + 1e-4
        x[i] -= push * nx
        y[i] -= push * ny
        x[j] += push * nx
        y[j] += push * ny

    return 1, impulse_events



@njit(cache=True)
def resolve_collisions_cells(
    x,
    y,
    vx,
    vy,
    ball_radius,
    restitution,
    grid_w,
    grid_h,
    cell_start,
    sorted_idx,
):
    pair_tests = 0
    overlaps = 0
    impulse_events = 0

    min_dist = ball_radius + ball_radius
    min_dist2 = min_dist * min_dist

    for cy in range(grid_h):
        row = cy * grid_w

        for cx in range(grid_w):
            cell = row + cx
            a0 = cell_start[cell]
            a1 = cell_start[cell + 1]

            for pa in range(a0, a1):
                i = sorted_idx[pa]
                for pb in range(pa + 1, a1):
                    j = sorted_idx[pb]
                    pair_tests += 1
                    ov, imp = resolve_pair(
                        x, y, vx, vy, i, j, min_dist, min_dist2, restitution
                    )
                    overlaps += ov
                    impulse_events += imp

            if cx + 1 < grid_w:
                east = cell + 1
                b0 = cell_start[east]
                b1 = cell_start[east + 1]
                for pa in range(a0, a1):
                    i = sorted_idx[pa]
                    for pb in range(b0, b1):
                        j = sorted_idx[pb]
                        pair_tests += 1
                        ov, imp = resolve_pair(
                            x, y, vx, vy, i, j, min_dist, min_dist2, restitution
                        )
                        overlaps += ov
                        impulse_events += imp

            if cy + 1 < grid_h:
                south_row = (cy + 1) * grid_w

                if cx > 0:
                    south_west = south_row + (cx - 1)
                    b0 = cell_start[south_west]
                    b1 = cell_start[south_west + 1]
                    for pa in range(a0, a1):
                        i = sorted_idx[pa]
                        for pb in range(b0, b1):
                            j = sorted_idx[pb]
                            pair_tests += 1
                            ov, imp = resolve_pair(
                                x, y, vx, vy, i, j, min_dist, min_dist2, restitution
                            )
                            overlaps += ov
                            impulse_events += imp

                south = south_row + cx
                b0 = cell_start[south]
                b1 = cell_start[south + 1]
                for pa in range(a0, a1):
                    i = sorted_idx[pa]
                    for pb in range(b0, b1):
                        j = sorted_idx[pb]
                        pair_tests += 1
                        ov, imp = resolve_pair(
                            x, y, vx, vy, i, j, min_dist, min_dist2, restitution
                        )
                        overlaps += ov
                        impulse_events += imp

                if cx + 1 < grid_w:
                    south_east = south_row + (cx + 1)
                    b0 = cell_start[south_east]
                    b1 = cell_start[south_east + 1]
                    for pa in range(a0, a1):
                        i = sorted_idx[pa]
                        for pb in range(b0, b1):
                            j = sorted_idx[pb]
                            pair_tests += 1
                            ov, imp = resolve_pair(
                                x, y, vx, vy, i, j, min_dist, min_dist2, restitution
                            )
                            overlaps += ov
                            impulse_events += imp

    return pair_tests, overlaps, impulse_events


@njit(cache=True, fastmath=True)
def enforce_min_speed(vx, vy, n, min_speed):
    speed_fixes = 0
    min_speed2 = min_speed * min_speed

    for i in range(n):
        s2 = vx[i] * vx[i] + vy[i] * vy[i]
        if 0.0 < s2 < min_speed2:
            s = math.sqrt(s2)
            scale = min_speed / s
            vx[i] *= scale
            vy[i] *= scale
            speed_fixes += 1

    return speed_fixes


@njit(cache=True)
def draw_points_rgb(frame_rgb, x, y, color_r, color_g, color_b, n):
    w = frame_rgb.shape[0]
    h = frame_rgb.shape[1]

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])

        if 0 <= xi < w and 0 <= yi < h:
            frame_rgb[xi, yi, 0] = color_r[i]
            frame_rgb[xi, yi, 1] = color_g[i]
            frame_rgb[xi, yi, 2] = color_b[i]


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------
def random_color():
    return (
        random.randint(80, 255),
        random.randint(80, 255),
        random.randint(80, 255),
    )


def add_ball(x, y, vx, vy, cr, cg, cb, n, px, py, pvx, pvy):
    if n >= x.shape[0]:
        return n

    x[n] = np.float32(px)
    y[n] = np.float32(py)
    vx[n] = np.float32(pvx)
    vy[n] = np.float32(pvy)

    r, g, b = random_color()
    cr[n] = r
    cg[n] = g
    cb[n] = b

    return n + 1


def find_spawn_position_grid(
    x,
    y,
    n,
    cell_start,
    sorted_idx,
    cx,
    cy,
    arena_radius,
    ball_radius,
    cell_size,
    grid_w,
    grid_h,
    trials=120,
):
    if n == 0:
        return (float(cx), float(cy)), 0

    min_sep = ball_radius + ball_radius + 0.25
    min_sep2 = min_sep * min_sep

    best_pos = None
    best_clearance = -1.0

    for trial in range(1, trials + 1):
        angle = random.uniform(0.0, 2.0 * math.pi)
        rr = (arena_radius - ball_radius - 2.0) * math.sqrt(random.random())
        px = cx + rr * math.cos(angle)
        py = cy + rr * math.sin(angle)

        gx = int(px / cell_size)
        gy = int(py / cell_size)

        if gx < 0:
            gx = 0
        elif gx >= grid_w:
            gx = grid_w - 1

        if gy < 0:
            gy = 0
        elif gy >= grid_h:
            gy = grid_h - 1

        valid = True
        local_min_d2 = 1.0e30

        for oy in range(-1, 2):
            ny = gy + oy
            if ny < 0 or ny >= grid_h:
                continue

            for ox in range(-1, 2):
                nx = gx + ox
                if nx < 0 or nx >= grid_w:
                    continue

                cell = ny * grid_w + nx
                a0 = cell_start[cell]
                a1 = cell_start[cell + 1]

                for p in range(a0, a1):
                    j = sorted_idx[p]
                    dx = px - float(x[j])
                    dy = py - float(y[j])
                    d2 = dx * dx + dy * dy

                    if d2 < min_sep2:
                        valid = False
                        break

                    if d2 < local_min_d2:
                        local_min_d2 = d2

                if not valid:
                    break

            if not valid:
                break

        if valid:
            clearance = math.sqrt(local_min_d2) if local_min_d2 < 1.0e29 else 1.0e15
            if clearance > best_clearance:
                best_clearance = clearance
                best_pos = (px, py)

    return best_pos, trials


def estimate_hex_max_balls(arena_radius, ball_radius):
    if ball_radius <= 0.0:
        return 0.0
    return HEX_PACKING_DENSITY * (arena_radius * arena_radius) / (ball_radius * ball_radius)


def estimate_physics_flops(
    integrate_particles,
    wall_hits,
    pair_tests,
    overlaps,
    minspeed_particles,
    speed_fixes,
):
    return (
        11.0 * integrate_particles
        + 20.0 * wall_hits
        + 5.0 * pair_tests
        + 30.0 * overlaps
        + 3.0 * minspeed_particles
        + 8.0 * speed_fixes
    )


def format_big(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}k"
    return str(n)


def hud_mode_name(mode):
    if mode == HUD_OFF:
        return "off"
    if mode == HUD_COUNT:
        return "count"
    if mode == HUD_BASIC:
        return "basic"
    return "full"


def build_hud_lines(
    hud_mode,
    n,
    total_spawned,
    total_spawn_trials,
    occupied_cells,
    avg_occ,
    max_cell_occ,
    fps_real,
    sim_speed,
    avg_physics_ms,
    avg_render_ms,
    pair_tests_per_frame,
    overlaps_per_frame,
    wall_hits_per_frame,
    impulses_per_frame,
    speed_fixes_per_frame,
    area_fill,
    count_saturation,
    est_flops_per_frame,
    est_gflops,
):
    count_lines = [
        f"Balls: {n:,} / {MAX_BALLS:,}",
    ]

    basic_lines = [
        (
            f"FPS(real): {fps_real:6.1f}   sim speed: {sim_speed:5.2f}x   "
            f"substeps: {SUBSTEPS}"
        ),
        (
            f"Physics: {avg_physics_ms:6.2f} ms/frame   "
            f"Render: {avg_render_ms:6.2f} ms/frame   "
            f"renderer: {'points' if FAST_POINT_RENDER else 'circles'}"
        ),
        (
            f"Grid: {occupied_cells:,}/{GRID_CELLS:,} occupied   "
            f"avg occ: {avg_occ:5.2f}   max occ: {max_cell_occ}"
        ),
        (
            f"Pair tests/frame: {format_big(int(pair_tests_per_frame))}   "
            f"overlaps/frame: {format_big(int(overlaps_per_frame))}   "
            f"wall hits/frame: {format_big(int(wall_hits_per_frame))}"
        ),
        (
            f"Impulses/frame: {format_big(int(impulses_per_frame))}   "
            f"speed fixes/frame: {format_big(int(speed_fixes_per_frame))}"
        ),
    ]

    full_lines = [
        (
            f"Ball area fill: {area_fill * 100.0:6.2f}%   "
            f"count vs hex upper bound: {count_saturation * 100.0:6.2f}%"
        ),
        (
            f"Est physics FLOPs/frame: {format_big(int(est_flops_per_frame))}   "
            f"Est GFLOP/s: {est_gflops:6.3f}"
        ),
        (
            f"Spawned: {total_spawned:,}   "
            f"spawn trials total: {total_spawn_trials:,}   "
            f"keys: H=HUD  R=render"
        ),
    ]

    if hud_mode == HUD_OFF:
        return []
    if hud_mode == HUD_COUNT:
        return count_lines
    if hud_mode == HUD_BASIC:
        return count_lines + basic_lines
    return count_lines + basic_lines + full_lines

@njit(cache=True)
def clamp_speed(vx, vy, n, max_speed):
    max_s2 = max_speed * max_speed
    for i in range(n):
        s2 = vx[i] * vx[i] + vy[i] * vy[i]
        if s2 > max_s2:
            s = math.sqrt(s2)
            scale = max_speed / s
            vx[i] *= scale
            vy[i] *= scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()

    caption = f"Ball Spawner [{hud_mode_name(HUD_BASIC)}]"
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    x = np.empty(MAX_BALLS, dtype=np.float32)
    y = np.empty(MAX_BALLS, dtype=np.float32)
    vx = np.empty(MAX_BALLS, dtype=np.float32)
    vy = np.empty(MAX_BALLS, dtype=np.float32)

    color_r = np.empty(MAX_BALLS, dtype=np.uint8)
    color_g = np.empty(MAX_BALLS, dtype=np.uint8)
    color_b = np.empty(MAX_BALLS, dtype=np.uint8)

    cell_id = np.empty(MAX_BALLS, dtype=np.int32)
    cell_count = np.empty(GRID_CELLS, dtype=np.int32)
    cell_start = np.empty(GRID_CELLS + 1, dtype=np.int32)
    cell_cursor = np.empty(GRID_CELLS, dtype=np.int32)
    sorted_idx = np.empty(MAX_BALLS, dtype=np.int32)

    background_surface = pygame.Surface((WIDTH, HEIGHT))
    background_surface.fill(BG_COLOR)
    pygame.draw.circle(
        background_surface,
        RING_COLOR,
        (CENTER_X, CENTER_Y),
        CIRCLE_RADIUS,
        2,
    )

    background_rgb = pygame.surfarray.array3d(background_surface)
    frame_rgb = np.empty_like(background_rgb)

    n = 0
    total_spawned = 0
    total_spawn_trials = 0

    for _ in range(2):
        a = random.uniform(0.0, 2.0 * math.pi)
        r = random.uniform(0.0, CIRCLE_RADIUS * 0.35)
        px = CENTER_X + r * math.cos(a)
        py = CENTER_Y + r * math.sin(a)

        speed = random.uniform(300.0, 480.0)
        va = random.uniform(0.0, 2.0 * math.pi)
        pvx = speed * math.cos(va)
        pvy = speed * math.sin(va)

        n = add_ball(x, y, vx, vy, color_r, color_g, color_b, n, px, py, pvx, pvy)

    integrate_and_wall(
        x,
        y,
        vx,
        vy,
        n,
        0.0,
        float(GRAVITY),
        float(CENTER_X),
        float(CENTER_Y),
        float(CIRCLE_RADIUS),
        float(BALL_RADIUS),
        float(WALL_RESTITUTION),
    )
    build_grid(
        x,
        y,
        n,
        CELL_SIZE,
        GRID_W,
        GRID_H,
        cell_id,
        cell_count,
        cell_start,
        cell_cursor,
        sorted_idx,
    )
    resolve_collisions_cells(
        x,
        y,
        vx,
        vy,
        float(BALL_RADIUS),
        float(BALL_RESTITUTION),
        GRID_W,
        GRID_H,
        cell_start,
        sorted_idx,
    )
    enforce_min_speed(vx, vy, n, float(MIN_SPEED))
    if FAST_POINT_RENDER:
        np.copyto(frame_rgb, background_rgb)
        draw_points_rgb(frame_rgb, x, y, color_r, color_g, color_b, n)

    spawn_timer = 0.0
    render_enabled = True
    hud_mode = HUD_BASIC
    running = True

    hud_lines = []
    hud_last_update = time.perf_counter()

    acc_frames = 0
    acc_sim_time = 0.0
    acc_physics_ms = 0.0
    acc_render_ms = 0.0

    acc_integrate_particles = 0
    acc_minspeed_particles = 0
    acc_wall_hits = 0
    acc_pair_tests = 0
    acc_overlaps = 0
    acc_impulses = 0
    acc_speed_fixes = 0

    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.025)
        spawn_timer -= dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    render_enabled = not render_enabled

                elif event.key == pygame.K_h:
                    hud_mode = (hud_mode + 1) % HUD_MODE_COUNT
                    pygame.display.set_caption(
                        f"Ball Spawner [{hud_mode_name(hud_mode)}]"
                    )

        physics_t0 = time.perf_counter()

        collision_happened = False
        wall_hits_frame = 0
        pair_tests_frame = 0
        overlaps_frame = 0
        impulses_frame = 0

        sub_dt = dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            wall_hits_frame += integrate_and_wall(
                x,
                y,
                vx,
                vy,
                n,
                float(sub_dt),
                float(GRAVITY),
                float(CENTER_X),
                float(CENTER_Y),
                float(CIRCLE_RADIUS),
                float(BALL_RADIUS),
                float(WALL_RESTITUTION),
            )

            build_grid(
                x,
                y,
                n,
                CELL_SIZE,
                GRID_W,
                GRID_H,
                cell_id,
                cell_count,
                cell_start,
                cell_cursor,
                sorted_idx,
            )

            pair_tests, overlaps, impulses = resolve_collisions_cells(
                x,
                y,
                vx,
                vy,
                float(BALL_RADIUS),
                float(BALL_RESTITUTION),
                GRID_W,
                GRID_H,
                cell_start,
                sorted_idx,
            )

            pair_tests_frame += pair_tests
            overlaps_frame += overlaps
            impulses_frame += impulses

            if overlaps > 0:
                collision_happened = True
        
        clamp_speed(vx, vy, n, float(MAX_SPEED))
        speed_fixes_frame = enforce_min_speed(vx, vy, n, float(MIN_SPEED))

        build_grid(
            x,
            y,
            n,
            CELL_SIZE,
            GRID_W,
            GRID_H,
            cell_id,
            cell_count,
            cell_start,
            cell_cursor,
            sorted_idx,
        )

        if collision_happened and spawn_timer <= 0.0 and n < MAX_BALLS:
            pos, trials_used = find_spawn_position_grid(
                x,
                y,
                n,
                cell_start,
                sorted_idx,
                CENTER_X,
                CENTER_Y,
                CIRCLE_RADIUS,
                BALL_RADIUS,
                CELL_SIZE,
                GRID_W,
                GRID_H,
                trials=120,
            )
            total_spawn_trials += trials_used

            if pos is not None:
                speed = random.uniform(180.0, 360.0)
                va = random.uniform(0.0, 2.0 * math.pi)
                pvx = speed * math.cos(va)
                pvy = speed * math.sin(va)

                n = add_ball(
                    x,
                    y,
                    vx,
                    vy,
                    color_r,
                    color_g,
                    color_b,
                    n,
                    pos[0],
                    pos[1],
                    pvx,
                    pvy,
                )
                total_spawned += 1
                spawn_timer = SPAWN_COOLDOWN

        physics_ms = (time.perf_counter() - physics_t0) * 1000.0

        render_t0 = time.perf_counter()

        if render_enabled:
            if FAST_POINT_RENDER:
                np.copyto(frame_rgb, background_rgb)
                draw_points_rgb(frame_rgb, x, y, color_r, color_g, color_b, n)
                pygame.surfarray.blit_array(screen, frame_rgb)
            else:
                screen.blit(background_surface, (0, 0))
                r_int = int(round(BALL_RADIUS))
                for i in range(n):
                    pygame.draw.circle(
                        screen,
                        (int(color_r[i]), int(color_g[i]), int(color_b[i])),
                        (int(x[i]), int(y[i])),
                        r_int,
                    )
        else:
            screen.fill(BG_COLOR)

        now = time.perf_counter()

        acc_frames += 1
        acc_sim_time += dt
        acc_physics_ms += physics_ms
        acc_render_ms += (time.perf_counter() - render_t0) * 1000.0

        acc_integrate_particles += n * SUBSTEPS
        acc_minspeed_particles += n
        acc_wall_hits += wall_hits_frame
        acc_pair_tests += pair_tests_frame
        acc_overlaps += overlaps_frame
        acc_impulses += impulses_frame
        acc_speed_fixes += speed_fixes_frame

        hud_elapsed = now - hud_last_update
        if hud_elapsed >= HUD_UPDATE_PERIOD:
            occupied_cells = int(np.count_nonzero(cell_count))
            max_cell_occ = int(cell_count.max()) if GRID_CELLS > 0 else 0
            avg_occ = (n / occupied_cells) if occupied_cells > 0 else 0.0

            area_fill = (n * BALL_RADIUS * BALL_RADIUS) / (CIRCLE_RADIUS * CIRCLE_RADIUS)
            est_max_balls = estimate_hex_max_balls(CIRCLE_RADIUS, BALL_RADIUS)
            count_saturation = (n / est_max_balls) if est_max_balls > 0.0 else 0.0

            est_flops = estimate_physics_flops(
                acc_integrate_particles,
                acc_wall_hits,
                acc_pair_tests,
                acc_overlaps,
                acc_minspeed_particles,
                acc_speed_fixes,
            )

            fps_real = acc_frames / hud_elapsed if hud_elapsed > 0.0 else 0.0
            sim_speed = acc_sim_time / hud_elapsed if hud_elapsed > 0.0 else 0.0
            avg_physics_ms = acc_physics_ms / max(acc_frames, 1)
            avg_render_ms = acc_render_ms / max(acc_frames, 1)

            est_flops_per_frame = est_flops / max(acc_frames, 1)
            est_gflops = est_flops / hud_elapsed / 1.0e9 if hud_elapsed > 0.0 else 0.0

            pair_tests_per_frame = acc_pair_tests / max(acc_frames, 1)
            overlaps_per_frame = acc_overlaps / max(acc_frames, 1)
            wall_hits_per_frame = acc_wall_hits / max(acc_frames, 1)
            impulses_per_frame = acc_impulses / max(acc_frames, 1)
            speed_fixes_per_frame = acc_speed_fixes / max(acc_frames, 1)

            hud_lines = build_hud_lines(
                hud_mode=hud_mode,
                n=n,
                total_spawned=total_spawned,
                total_spawn_trials=total_spawn_trials,
                occupied_cells=occupied_cells,
                avg_occ=avg_occ,
                max_cell_occ=max_cell_occ,
                fps_real=fps_real,
                sim_speed=sim_speed,
                avg_physics_ms=avg_physics_ms,
                avg_render_ms=avg_render_ms,
                pair_tests_per_frame=pair_tests_per_frame,
                overlaps_per_frame=overlaps_per_frame,
                wall_hits_per_frame=wall_hits_per_frame,
                impulses_per_frame=impulses_per_frame,
                speed_fixes_per_frame=speed_fixes_per_frame,
                area_fill=area_fill,
                count_saturation=count_saturation,
                est_flops_per_frame=est_flops_per_frame,
                est_gflops=est_gflops,
            )

            acc_frames = 0
            acc_sim_time = 0.0
            acc_physics_ms = 0.0
            acc_render_ms = 0.0

            acc_integrate_particles = 0
            acc_minspeed_particles = 0
            acc_wall_hits = 0
            acc_pair_tests = 0
            acc_overlaps = 0
            acc_impulses = 0
            acc_speed_fixes = 0

            hud_last_update = now

        if hud_mode != HUD_OFF:
            y_px = 10
            for line in hud_lines:
                surf = font.render(line, True, TEXT_COLOR)
                screen.blit(surf, (10, y_px))
                y_px += 20

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
