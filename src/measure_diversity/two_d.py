import math
from typing import List, Tuple

# def create_uniform_normed_datapoints(n: int = 10) -> List[Tuple[float, float]]:
#     """
#     Create n equally spaced points on the unit circle (2D).
#
#     Args:
#         n: Number of points to generate (>= 1)
#
#     Returns:
#         List of (x, y) tuples, each lying on the unit circle.
#     """
#     if n < 1:
#         raise ValueError("n must be >= 1")
#
#     points = []
#     for i in range(n):
#         angle = 2 * math.pi * i / n
#         x = math.cos(angle)
#         y = math.sin(angle)
#         points.append((x, y))
#     return points


# def create_firstquad_normed_datapoints(n: int = 10) -> List[Tuple[float, float]]:
#     """
#     Create n equally spaced points on the unit circle (2D) in the first quadrant.
#
#     The first quadrant is defined as 0 <= angle <= π/2.
#
#     Args:
#         n: Number of points to generate (>= 1)
#
#     Returns:
#         List of (x, y) tuples in the first quadrant of the unit circle.
#     """
#     if n < 1:
#         raise ValueError("n must be >= 1")
#
#     points = []
#     for i in range(n):
#         angle = (math.pi / 2) * i / (n - 1) if n > 1 else 0
#         x = math.cos(angle)
#         y = math.sin(angle)
#         points.append((x, y))
#     return points


def create_normed_datapoints(n: int = 8, quads: list[int] | None = None) -> List[Tuple[float, float]]:
    """
    Create n equally spaced points on the unit circle in the given quadrants.

    Quadrants:
        1: 0 to π/2
        2: π/2 to π
        3: π to 3π/2
        4: 3π/2 to 2π

    Args:
        n: Number of points to generate (>= 1)
        quads: List of quadrants to include, each in {1, 2, 3, 4}.
               If None or empty, distribute points equally on the full circle.

    Returns:
        List of (x, y) tuples.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Map quadrants to (start_angle, end_angle)
    quad_ranges = {
        1: (0, math.pi / 2),
        2: (math.pi / 2, math.pi),
        3: (math.pi, 3 * math.pi / 2),
        4: (3 * math.pi / 2, 2 * math.pi),
    }

    if not quads:
        # Default: full circle
        total_span = 2 * math.pi
        start_end_pairs = [(0, 2 * math.pi)]
    else:
        if not all(q in {1, 2, 3, 4} for q in quads):
            raise ValueError("quads must only contain values in {1, 2, 3, 4}")
        start_end_pairs = [quad_ranges[q] for q in quads]
        total_span = sum(end - start for start, end in start_end_pairs)

    points = []
    remaining = n
    for start, end in start_end_pairs:
        segment_span = end - start
        num_points_here = round(n * (segment_span / total_span))
        if num_points_here > remaining:
            num_points_here = remaining
        if num_points_here <= 0:
            continue

        for i in range(num_points_here):
            angle = start + (segment_span * i / (num_points_here - 1)) if num_points_here > 1 else start
            x = math.cos(angle)
            y = math.sin(angle)
            points.append((x, y))

        remaining -= num_points_here

    # Adjust count if rounding was imperfect
    if len(points) < n:
        start, end = start_end_pairs[0]
        for i in range(n - len(points)):
            angle = start + (end - start) * (i + 1) / (n + 1)
            points.append((math.cos(angle), math.sin(angle)))
    elif len(points) > n:
        points = points[:n]

    return points


def plot_points(points: List[Tuple[float, float]], title: str = "Points on the Unit Circle") -> None:
    """
    Plot a list of 2D points using matplotlib.

    Args:
        points: List of (x, y) tuples.
        title: Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    xs, ys = zip(*points)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c="red")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)
    plt.show()