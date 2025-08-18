from measure_diversity.measure import distance_dispersion, mean_pairwise_distance, cluster_inertia_diversity, convex_hull_volume
from measure_diversity import two_d
import pytest
import numpy as np

# def test_pairwise_diversity():
#     assert mean_pairwise_distance([[1, 0]]) == 0.0  # Single point, no pairs
#     assert mean_pairwise_distance([[1, 0], [0, 1]]) == 1.0  # Two orthogonal points
#     assert mean_pairwise_distance([[1, 0], [1, 0], [0, 1]]) == 2 / 3  # =(1+1+0)/3
#     assert mean_pairwise_distance([[0.5, 0.5], [-0.5, -0.5]]) == mean_pairwise_distance([[1, 1], [-1, -1]])
#
#     # test
#     points_equi = two_d.create_normed_datapoints(n=8)
#     points_first_quad = two_d.create_normed_datapoints(n=8, quads=[1])
#     points_first_and_third_quad = two_d.create_normed_datapoints(n=8, quads=[1, 3])
#     points_second_quad = two_d.create_normed_datapoints(n=8, quads=[2])
#
#     assert mean_pairwise_distance(points_equi) > mean_pairwise_distance(points_first_and_third_quad)
#     assert mean_pairwise_distance(points_equi) > mean_pairwise_distance(points_first_quad)
#     assert mean_pairwise_distance(points_first_and_third_quad) > mean_pairwise_distance(points_first_quad)
#     assert mean_pairwise_distance(points_first_quad) == mean_pairwise_distance(points_second_quad)


class TestDistanceFunctions:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError for all functions."""
        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            mean_pairwise_distance([])

        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            distance_dispersion([])

        with pytest.raises(ValueError, match="Cannot compute convex hull for empty data"):
            convex_hull_volume([])

        with pytest.raises(ValueError, match="Cannot compute cluster inertia for empty data"):
            cluster_inertia_diversity([])

    def test_single_datapoint_raises_error(self):
        """Test that single datapoint raises ValueError for both functions."""
        single_point = [[1, 2, 3]]

        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            mean_pairwise_distance(single_point)

        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            distance_dispersion(single_point)

        # Geometric functions should return 0.0 for single point
        assert convex_hull_volume(single_point) == 0.0
        assert cluster_inertia_diversity(single_point) == 0.0

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]

        # For two points, mean and sum should be the same
        mean_dist = mean_pairwise_distance(data, metric="euclidean")
        sum_dist = distance_dispersion(data, metric="euclidean")

        expected = np.sqrt(2)  # sqrt((1-0)^2 + (1-0)^2)

        # same result between mean and sum as there is only one distance calculated
        assert np.isclose(mean_dist, expected)
        assert np.isclose(sum_dist, expected)

    def test_three_points_euclidean(self):
        """Test with three points using euclidean distance."""
        data = [[0, 0], [1, 0], [0, 1]]

        # Manual calculation:
        # dist(p0,p1) = 1, dist(p0,p2) = 1, dist(p1,p2) = sqrt(2)
        expected_distances = [1.0, 1.0, np.sqrt(2)]
        expected_mean = np.mean(expected_distances)
        expected_sum = np.sum(expected_distances)

        mean_dist = mean_pairwise_distance(data, metric="euclidean")
        sum_dist = distance_dispersion(data, metric="euclidean")

        assert np.isclose(mean_dist, expected_mean)
        assert np.isclose(sum_dist, expected_sum)

    def test_different_metrics(self):
        """Test that different metrics produce different results."""
        data = [[1, 2], [3, 4], [5, 6]]

        euclidean_mean = mean_pairwise_distance(data, metric="euclidean")
        cosine_mean = mean_pairwise_distance(data, metric="cosine")
        manhattan_mean = mean_pairwise_distance(data, metric="cityblock")

        # Results should be different for different metrics
        assert not np.isclose(euclidean_mean, cosine_mean)
        assert not np.isclose(euclidean_mean, manhattan_mean)
        assert not np.isclose(cosine_mean, manhattan_mean)

    def test_custom_metric_function(self):
        """Test using a custom metric function."""

        def custom_metric(u, v):
            """Simple L1 distance (Manhattan)"""
            return np.sum(np.abs(u - v))

        data = [[0, 0], [1, 1], [2, 2]]

        custom_mean = mean_pairwise_distance(data, metric=custom_metric)
        manhattan_mean = mean_pairwise_distance(data, metric="cityblock")

        # Should be the same since our custom metric is Manhattan distance
        assert np.isclose(custom_mean, manhattan_mean)