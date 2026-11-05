"""
Path planning playground  (pygame-ce + numpy)

Install:  pip install pygame-ce numpy

What this gives you
-------------------
* A grid map you can paint walls on with the mouse, plus a draggable start and goal.
* A Planner class that is a deliberately dumb stub (a straight line that ignores walls).
  YOUR JOB: replace its guts with BFS / Dijkstra / greedy / A* / RRT / whatever you like.
* A checker that tells you if your path is valid (hits a wall? gaps? illegal diagonal?)
  and what its cost is, so you can compare algorithms and heuristics objectively.
* An animation of the cells your planner explored, in the order it explored them.

Conventions (the classic gotcha, read this once):
    cell  = (x, y) = (column, row)
    grid  = numpy bool array, indexed grid[y, x]   (row first!)   True = wall
    cost  = 1 for an orthogonal step, sqrt(2) for a diagonal step

Controls
--------
Left-drag   paint walls (or drag the green start / red goal)
Right-drag  erase walls
SPACE       replan and replay the exploration animation
D           toggle 4-way / 8-way moves
N           random walls       C  clear walls       P  preset map (wall with a gap + a trap)
+ / -       animation speed    ESC quit
"""

import time
import traceback

import numpy as np
import pygame

COLS, ROWS = 52, 32
CELL = 18
HUD_H = 78
W, H = COLS * CELL, ROWS * CELL + HUD_H
SQRT2 = 2 ** 0.5
rng = np.random.default_rng()


# --------------------------------------------------------------------------
# Helpers (use these, or ignore them and write your own)
# --------------------------------------------------------------------------
def in_bounds(grid, cell):
    x, y = cell
    return 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]


def is_free(grid, cell):
    return in_bounds(grid, cell) and not grid[cell[1], cell[0]]


def neighbors(grid, cell, diagonal=True):
    """Yield (neighbor_cell, step_cost) for every legal move from `cell`.
    Diagonal moves are not allowed to cut the corner of a wall."""
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (x + dx, y + dy)
        if is_free(grid, n):
            yield n, 1.0
    if diagonal:
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (x + dx, y + dy)
            if is_free(grid, n) and is_free(grid, (x + dx, y)) and is_free(grid, (x, y + dy)):
                yield n, SQRT2


def straight_line(a, b):
    """Bresenham line from cell a to cell b (inclusive). Ignores walls."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    pts = []
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def check_path(grid, path, start, goal, diagonal):
    """Returns (is_valid, message, cost)."""
    if not path:
        return False, "no path returned", 0.0
    if path[0] != start or path[-1] != goal:
        return False, "path must run from start to goal", 0.0
    cost = 0.0
    for a, b in zip(path, path[1:]):
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if max(dx, dy) != 1:
            return False, f"gap between {a} and {b}", 0.0
        if not is_free(grid, b):
            return False, f"path hits a wall at {b}", 0.0
        if dx and dy:
            if not diagonal:
                return False, f"diagonal step {a} -> {b} in 4-way mode", 0.0
            if not (is_free(grid, (b[0], a[1])) and is_free(grid, (a[0], b[1]))):
                return False, f"step {a} -> {b} cuts a wall corner", 0.0
            cost += SQRT2
        else:
            cost += 1.0
    return True, "valid", cost


# --------------------------------------------------------------------------
# YOUR CODE GOES HERE
# --------------------------------------------------------------------------
class Planner:
    """
    Stub planner: draws a straight line and ignores every wall. Replace it.

    The sim calls:
        plan(grid, start, goal, diagonal)
            grid      numpy bool array [ROWS, COLS], True = wall (it's a copy, scribble freely)
            start     (x, y) cell
            goal      (x, y) cell
            diagonal  True for 8-way moves, False for 4-way
        It must return the path as a list of cells from start to goal inclusive,
        or an empty list / None if you think there is no path.

    The display reads these (all optional, but they make debugging far nicer):
        self.explored   cells in the order your algorithm expanded/visited them (drives the animation)
        self.frontier   cells still waiting in your open set when you finished
        self.edges      list of ((x0, y0), (x1, y1)) pairs, e.g. the tree for RRT

    Things to try, roughly in order:
        1. BFS on the 4-way grid. Check it finds a valid path around the wall.
        2. Dijkstra with the step costs from neighbors() (heapq is your friend). Switch to 8-way.
        3. Greedy best-first (priority = heuristic only). Watch it fall into the cup, then compare cost.
        4. A*: priority = cost_so_far + heuristic. Try Manhattan vs Euclidean vs octile:
           which are admissible in 8-way mode? Which one expands the fewest cells?
        5. Smooth the path: remove waypoints you can skip using a line-of-sight check.
        6. Paint walls while it runs: how does your planner cope with the goal being unreachable?
        7. RRT: sample random cells, grow a tree (fill self.edges), and see why paths look wobbly.
    """

    def __init__(self):
        self.reset_debug()

    def reset_debug(self):
        self.explored = []
        self.frontier = []
        self.edges = []

    def plan(self, grid, start, goal, diagonal):
        self.reset_debug()

        # TODO: replace this with a real planner
        path = straight_line(start, goal)
        self.explored = list(path)
        return path


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------
class World:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=bool)
        self.diagonal = True
        self.start = (4, ROWS // 2)
        self.goal = (COLS - 5, ROWS // 2)
        self.preset()

    def clear(self):
        self.grid[:] = False

    def preset(self):
        self.clear()
        self.start = (4, ROWS // 2)
        self.goal = (COLS - 5, ROWS // 2)
        self.grid[0:ROWS - 8, COLS // 2] = True            # long wall with a gap at the bottom
        cx0, cx1 = 12, 19                                  # a cup opening towards the start:
        cy0, cy1 = ROWS // 2 - 5, ROWS // 2 + 5            # a trap for greedy planners
        self.grid[cy0, cx0:cx1 + 1] = True
        self.grid[cy1, cx0:cx1 + 1] = True
        self.grid[cy0:cy1 + 1, cx1] = True

    def randomise(self, density=0.22):
        self.grid[:] = rng.random((ROWS, COLS)) < density
        for x, y in (self.start, self.goal):
            self.grid[y, x] = False

    def apply(self, mode, cell):
        """Edit the world. Returns True if something changed."""
        x, y = cell
        if mode == "wall":
            if cell in (self.start, self.goal) or self.grid[y, x]:
                return False
            self.grid[y, x] = True
            return True
        if mode == "erase":
            if not self.grid[y, x]:
                return False
            self.grid[y, x] = False
            return True
        if mode == "start":
            if cell in (self.start, self.goal):
                return False
            self.start = cell
            self.grid[y, x] = False
            return True
        if mode == "goal":
            if cell in (self.start, self.goal):
                return False
            self.goal = cell
            self.grid[y, x] = False
            return True
        return False


def run_planner(planner, world):
    t0 = time.perf_counter()
    err = None
    try:
        path = planner.plan(world.grid.copy(), world.start, world.goal, world.diagonal)
    except Exception:
        traceback.print_exc()
        path, err = None, "planner raised an exception (see console)"
    ms = (time.perf_counter() - t0) * 1000.0

    path = [tuple(int(v) for v in c) for c in path] if path is not None and len(path) else []
    if err:
        valid, msg, cost = False, err, 0.0
    else:
        valid, msg, cost = check_path(world.grid, path, world.start, world.goal, world.diagonal)
    return dict(
        path=path, valid=valid, msg=msg, cost=cost, ms=ms,
        explored=[tuple(int(v) for v in c) for c in planner.explored],
        frontier=[tuple(int(v) for v in c) for c in planner.frontier],
        edges=list(planner.edges),
    )


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
BG = (14, 16, 22)
C_GRIDLINE = (24, 27, 36)
C_WALL = (70, 76, 92)
C_EXPLORED = (36, 52, 84)
C_FRONTIER = (170, 120, 50)
C_EDGE = (70, 110, 170)
C_PATH = (255, 210, 70)
C_START = (80, 220, 120)
C_GOAL = (240, 90, 90)
C_TEXT = (200, 205, 215)
C_BAD = (240, 110, 110)

SPEEDS = [1, 2, 5, 10, 25, 50, 100, 250]


def center(c):
    return (c[0] * CELL + CELL // 2, c[1] * CELL + CELL // 2)


def cell_at(pos):
    x, y = pos
    if x < 0 or y < 0 or x >= COLS * CELL or y >= ROWS * CELL:
        return None
    return (x // CELL, y // CELL)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Path planning playground")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    world = World()
    planner = Planner()
    res = run_planner(planner, world)
    reveal = 0.0
    speed_i = 3
    drag = None
    last = None
    running = True

    while running:
        dirty = False
        replay = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    replay = True
                elif k == pygame.K_d:
                    world.diagonal = not world.diagonal
                    dirty = True
                elif k == pygame.K_n:
                    world.randomise()
                    dirty = True
                elif k == pygame.K_c:
                    world.clear()
                    dirty = True
                elif k == pygame.K_p:
                    world.preset()
                    dirty = True
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed_i = min(len(SPEEDS) - 1, speed_i + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_i = max(0, speed_i - 1)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                c = cell_at(e.pos)
                if c:
                    if e.button == 1 and c == world.start:
                        drag = "start"
                    elif e.button == 1 and c == world.goal:
                        drag = "goal"
                    elif e.button == 1:
                        drag = "wall"
                    else:
                        drag = "erase"
                    last = c
                    dirty |= world.apply(drag, c)
            elif e.type == pygame.MOUSEMOTION and drag:
                c = cell_at(e.pos)
                if c and c != last:
                    for p in straight_line(last, c):     # fill the gaps if the mouse moved fast
                        dirty |= world.apply(drag, p)
                    last = c
            elif e.type == pygame.MOUSEBUTTONUP and e.button in (1, 3):
                drag = None

        if replay:
            res = run_planner(planner, world)
            reveal = 0.0
        elif dirty:
            res = run_planner(planner, world)
            reveal = float(len(res["explored"]))     # show the result instantly while editing

        n_explored = len(res["explored"])
        if reveal < n_explored:
            reveal = min(n_explored, reveal + SPEEDS[speed_i])
        done = reveal >= n_explored
        shown = int(reveal)

        # ---------------- draw ----------------
        screen.fill(BG)

        for c in res["explored"][:shown]:
            pygame.draw.rect(screen, C_EXPLORED, (c[0] * CELL, c[1] * CELL, CELL, CELL))
        if done:
            for c in res["frontier"]:
                pygame.draw.rect(screen, C_FRONTIER, (c[0] * CELL, c[1] * CELL, CELL, CELL))

        ys, xs = np.nonzero(world.grid)
        for x, y in zip(xs.tolist(), ys.tolist()):
            pygame.draw.rect(screen, C_WALL, (x * CELL, y * CELL, CELL, CELL))

        for x in range(COLS + 1):
            pygame.draw.line(screen, C_GRIDLINE, (x * CELL, 0), (x * CELL, ROWS * CELL))
        for y in range(ROWS + 1):
            pygame.draw.line(screen, C_GRIDLINE, (0, y * CELL), (COLS * CELL, y * CELL))

        edges = res["edges"]
        if edges:
            frac = 1.0 if n_explored == 0 else min(1.0, reveal / n_explored)
            for a, b in edges[:int(len(edges) * frac)]:
                pygame.draw.line(screen, C_EDGE, center(a), center(b), 1)

        if done and len(res["path"]) > 1:
            pygame.draw.lines(screen, C_PATH, False, [center(c) for c in res["path"]], 3)

        pygame.draw.circle(screen, C_START, center(world.start), CELL // 2 - 1)
        pygame.draw.circle(screen, C_GOAL, center(world.goal), CELL // 2 - 1)

        # HUD
        pygame.draw.rect(screen, (20, 23, 31), (0, ROWS * CELL, W, HUD_H))
        if res["valid"]:
            status = f"path OK   cost {res['cost']:.2f}   {len(res['path'])} cells"
            col = C_START
        else:
            status = f"path INVALID: {res['msg']}"
            col = C_BAD
        hud = [
            (status, col),
            (f"expanded {n_explored}   time {res['ms']:.2f} ms   moves {'8-way' if world.diagonal else '4-way'} (D)"
             f"   animation {SPEEDS[speed_i]} cells/frame (+/-)", C_TEXT),
            ("L-drag walls / start / goal   R-drag erase   SPACE replay   N random   C clear   P preset", C_TEXT),
        ]
        for i, (s, c) in enumerate(hud):
            screen.blit(font.render(s, True, c), (10, ROWS * CELL + 8 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
