from measure_diversity import evaluate_measures
from measure_diversity.measure import (mean_pairwise_distance, distance_dispersion, cluster_inertia_diversity,
                                       convex_hull_volume)
import numpy as np
from measure_diversity import two_d

def test_eval_measures_not_normed():
    """
    Test the evaluate_measures function to ensure it correctly evaluates
    diversity measures on a set of datasets.
    """
    # Run the evaluation
    clustered = np.random.normal([0, 0], 0.1, (20, 2)).tolist()
    medium = np.random.normal([0, 0], 1.0, (20, 2)).tolist()

    # get the maximum and minimum values for x and y
    all_points = np.vstack([clustered, medium])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    two_d.plot_points(clustered, title="Clustered Points", xlim=(x_min-0.1, x_max+0.1), ylim=(y_min-0.1, y_max+0.1))
    two_d.plot_points(medium, title="Medium Spread Points", xlim=(x_min-0.1, x_max+0.1), ylim=(y_min-0.1, y_max+0.1))

    measures = [
        lambda data: mean_pairwise_distance(data, metric="euclidean"),
        lambda data: mean_pairwise_distance(data, metric="cosine"),
        lambda data: distance_dispersion(data, metric="cosine"),
        convex_hull_volume,
        cluster_inertia_diversity
    ]

    # Add names to lambda functions for better display
    measures[0].__name__ = "mean_pairwise_euclidean"
    measures[1].__name__ = "mean_pairwise_cosine"
    measures[2].__name__ = "distance_dispersion_cosine"

    evaluate_measures.evaluate_monotone_order([clustered, medium], measures,
                                              dataset_names=["Clustered Spread", "Medium Spread"])

def test_normed():
    points_equi = two_d.create_normed_datapoints(n=8)
    points_first_quad = two_d.create_normed_datapoints(n=8, quads=[1])
    points_first_and_third_quad = two_d.create_normed_datapoints(n=8, quads=[1, 3])

    all_points = points_equi + points_first_quad + points_first_and_third_quad # + points_second_quad
    all_points = np.array(all_points)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    two_d.plot_points(points_equi, title="8 Uniform Points Around Circle",
                      xlim=(x_min-0.1, x_max+0.1), ylim=(y_min-0.1, y_max+0.1))
    two_d.plot_points(points_first_and_third_quad, title="8 Points in First and Third Quadrants of Circle",
                      xlim=(x_min-0.1, x_max+0.1), ylim=(y_min-0.1, y_max+0.1))
    two_d.plot_points(points_first_quad, title="8 Points in First Quadrant of Circle",
                      xlim=(x_min-0.1, x_max+0.1), ylim=(y_min-0.1, y_max+0.1))

    measures = [
        lambda data: mean_pairwise_distance(data, metric="euclidean"),
        lambda data: mean_pairwise_distance(data, metric="cosine"),
        lambda data: distance_dispersion(data, metric="cosine"),
        convex_hull_volume,
        cluster_inertia_diversity
    ]
    # Add names to lambda functions for better display
    measures[0].__name__ = "mean_pairwise_euclidean"
    measures[1].__name__ = "mean_pairwise_cosine"
    measures[2].__name__ = "distance_dispersion_cosine"
    evaluate_measures.evaluate_monotone_order(
        [points_first_quad, points_first_and_third_quad, points_equi],
        measures, dataset_names=["first_quad", "first_and_third_quad", "equi"],
    )