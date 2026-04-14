import random
import math
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


def duplicate_dataset(points_orig: list[Tuple[float, float]],
                      duplicate_int: Optional[list[int]] = None,
                      near_duplicates: bool = False,
                      perturb_lambda: float = 0.01) -> list[Tuple[float, float]]:
    """
    Duplicate each point in the dataset duplicate_int[index] times, i.e., 
    according to the corresponding value in duplicate_int. 
    Also supports near duplicates (small random perturbations of 
    the original data points)

    Parameters:
        points_orig (list[Tuple[float, float]]): Original dataset.
        duplicate_int (list[int], optional): List of integers indicating how many
                                   duplicates to add for each point. Should be the
                                   same length as points_orig. If None, all points
                                   are duplicated once.
        near_duplicates (bool, optional): If True, perturb duplicates.
        perturb_lambda (float, optional): Magnitude of perturbation.

    Returns:
        list[Tuple[float, float]]: New list of (x, y) coordinates
    """
    # if not provided, all points are duplicated once
    if duplicate_int is None:
        duplicate_int = [1] * len(points_orig)

    if len(points_orig) != len(duplicate_int):
        raise ValueError("points_orig and duplicate_int must have the same length")


    result = []
    for point, num_duplicates in zip(points_orig, duplicate_int):
        # Add the original
        result.append(point)

        # add the duplicates
        for _ in range(num_duplicates):
            if near_duplicates:
                x, y = point
                result.append((
                    x + random.uniform(-perturb_lambda, perturb_lambda),
                    y + random.uniform(-perturb_lambda, perturb_lambda)))
            else:
                result.append(point)
    return result


def create_toy_dataset1_axioms_challenges(num_points: int) -> Tuple[list[Tuple[float, float]], list[Tuple[float, float]]]:
    """
    Create the toy dataset from the 'Average and SumAverage' of
    the axioms and challenges paper (Mironov and Prokhorenkova, ICML 2025), 
    https://openreview.net/forum?id=2pdFMgv54m.
    Parameters:
        num_points (int): Number of points to generate

    Returns:
        Tuple[list[Tuple[float, float]], list[Tuple[float, float]]]: 
            A tuple containing two datasets:
            - The first list is the low diversity dataset (points at corners).
            - The second list is the high diversity dataset (points evenly distributed).
    """
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    # Dataset 1 (lower diversity):
    # all points at the corners of a unit square
    num_each = math.floor(num_points / 4)

    dataset_low_div = []
    for corner in corners:
        dataset_low_div.extend([corner] * num_each)

    # Add any remainder points
    remainder = num_points - len(dataset_low_div)
    dataset_low_div.extend(corners[:remainder])

    # Dataset 2 (higher diversity)
    # All points are evenly distributed across space

    # Determine the steps
    n_side = math.ceil(math.sqrt(num_points))
    
    # avoid division by zero
    if n_side == 1:
        coords = [0.0]
    else:
        coords = [i / (n_side - 1) for i in range(n_side)]

    # Generate points
    dataset_high_div = [(x, y) for x in coords for y in coords]

    # Trim to the specific number of points
    dataset_high_div = dataset_high_div[:num_points]

    return dataset_low_div, dataset_high_div
