import numpy as np
from typing import List, Tuple, Optional


def create_normed_datapoints(n: int = 8, quads: list[int] | None = None) -> List[Tuple[float, float]]:
    """
    Distributes n datapoints equidistantly on the unit circle (Einheitskreis)
    within the specified quadrants.

    Parameters:
    n (int): Number of datapoints to create (default: 8)
    quads (list[int] | None): List of quadrants (1, 2, 3, 4) where points should be distributed.
                              If None, defaults to all quadrants [1, 2, 3, 4]

    Returns:
    List[Tuple[float, float]]: List of (x, y) coordinate tuples

    Quadrants:
    1: [0, π/2]     (0° to 90°)
    2: [π/2, π]     (90° to 180°)
    3: [π, 3π/2]    (180° to 270°)
    4: [3π/2, 2π]   (270° to 360°)
    """

    # Default to all quadrants if none specified
    if quads is None:
        quads = [1, 2, 3, 4]

    if not quads:
        raise ValueError("At least one quadrant must be specified")

    # Sort quadrants for consistent behavior
    quads = sorted(set(quads))

    # Define quadrant angle ranges
    quad_ranges = {
        1: (0, np.pi / 2),
        2: (np.pi / 2, np.pi),
        3: (np.pi, 3 * np.pi / 2),
        4: (3 * np.pi / 2, 2 * np.pi)
    }

    # Special case: if all quadrants are selected, distribute over full circle
    if len(quads) == 4 and quads == [1, 2, 3, 4]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    else:
        # Calculate total angular span
        total_span = 0
        angle_segments = []

        for quad in quads:
            start_angle, end_angle = quad_ranges[quad]
            span = end_angle - start_angle
            total_span += span
            angle_segments.append((start_angle, end_angle, span))

        # Distribute points proportionally across segments
        angles = []

        for start_angle, end_angle, span in angle_segments:
            # Number of points for this segment (proportional to its span)
            points_in_segment = int(np.round(n * span / total_span))

            # Generate equidistant points in this segment
            if points_in_segment > 0:
                segment_angles = np.linspace(start_angle, end_angle,
                                             points_in_segment, endpoint=False)
                angles.extend(segment_angles)

        # If we have fewer points due to rounding, add the remaining ones
        while len(angles) < n:
            # Add points to the largest segment
            largest_segment_idx = np.argmax([seg[2] for seg in angle_segments])
            start_angle, end_angle, _ = angle_segments[largest_segment_idx]
            # Add a point at a reasonable position
            extra_angle = start_angle + (end_angle - start_angle) * (len(angles) - sum(
                int(np.round(n * seg[2] / total_span)) for seg in angle_segments[:largest_segment_idx + 1])) / (
                                      n - sum(int(np.round(n * seg[2] / total_span)) for seg in angle_segments))
            angles.append(extra_angle)

        # If we have too many points, remove some
        angles = angles[:n]
        angles = np.array(angles)

    # Convert to Cartesian coordinates on unit circle
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)

    # Return as list of (x, y) tuples
    points = [(float(x), float(y)) for x, y in zip(x_coords, y_coords)]

    return points


def plot_points(points: List[Tuple[float, float]],
                title: str = "Points on the Unit Circle",
                xlim: Optional[Tuple[float, float]] = None,
                ylim: Optional[Tuple[float, float]] = None
                ) -> None:
    """
    Plot a list of 2D points using matplotlib.

    Args:
        points: List of (x, y) tuples.
        title: Plot title.
        xlim: X-axis limits as (min, max). If None, uses matplotlib's default.
        ylim: Y-axis limits as (min, max). If None, uses matplotlib's default.
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

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.show()