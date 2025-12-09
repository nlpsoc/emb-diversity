from measure_diversity.measure import distance_dispersion, mean_pairwise_distance, cluster_inertia_diversity, \
    convex_hull_volume, diameter, bottleneck, energy
import pytest
import numpy as np


class TestMeanPairwiseDistance:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            mean_pairwise_distance([])

    def test_single_datapoint_raises_error(self):
        """Test that single datapoint raises ValueError."""
        single_point = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            mean_pairwise_distance(single_point)

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = mean_pairwise_distance(data)
        assert isinstance(result, float)

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        mean_dist = mean_pairwise_distance(data, metric="euclidean")
        expected = np.sqrt(2)  # sqrt((1-0)^2 + (1-0)^2)
        assert np.isclose(mean_dist, expected)

    def test_three_points_euclidean(self):
        """Test with three points using euclidean distance."""
        data = [[0, 0], [1, 0], [0, 1]]
        # Manual calculation: dist(p0,p1) = 1, dist(p0,p2) = 1, dist(p1,p2) = sqrt(2)
        expected_distances = [1.0, 1.0, np.sqrt(2)]
        expected_mean = np.mean(expected_distances)

        mean_dist = mean_pairwise_distance(data, metric="euclidean")
        assert np.isclose(mean_dist, expected_mean)

    def test_different_metrics(self):
        """Test that different metrics produce different results."""
        data = [[1, 2], [3, 4], [5, 6]]

        euclidean_mean = mean_pairwise_distance(data, metric="euclidean")
        cosine_mean = mean_pairwise_distance(data, metric="cosine")
        manhattan_mean = mean_pairwise_distance(data, metric="cityblock")  # cityblock is Manhattan distance

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

    def test_known_mathematical_values(self):
        """Test specific cases with known expected values."""
        # Two orthogonal unit vectors in cosine space
        assert np.isclose(mean_pairwise_distance([[1, 0], [0, 1]], metric="cosine"), 1.0)

        # Three points: two identical, one different - Distances: (1,1,0) -> mean = 2/3
        assert np.isclose(mean_pairwise_distance([[1, 0], [1, 0], [0, 1]], metric="cosine"), 2 / 3)

        # Scale invariance for cosine distance
        small_vectors = [[0.5, 0.5], [-0.5, -0.5]]
        large_vectors = [[1, 1], [-1, -1]]
        assert np.isclose(
            mean_pairwise_distance(small_vectors, metric="cosine"),
            mean_pairwise_distance(large_vectors, metric="cosine")
        )


class TestDiameter:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            diameter([])

    def test_single_datapoint_raises_error(self):
        """Test that single datapoint raises ValueError."""
        single_point = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            diameter(single_point)

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = diameter(data)
        assert isinstance(result, float)

    def test_three_points(self):
        # cosine distances

        # cosine distance between the first two points is 1 (as they are orthogonal)
        # so cosine similarity is 0 --> distance = 1
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = diameter(data)
        assert result == 1.0

        # vectors are all pointing in the same direction,
        # so cosine sim = 1 --> distance = 0
        data = [[1, 0], [3, 0], [5, 0]]
        result = diameter(data)
        assert result == 0.0

        # duplicates
        data = [[-1, 0], [-1, 0], [-1, 0]]
        result = diameter(data)
        assert result == 0.0


class TestBottleneck:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            bottleneck([])

    def test_single_datapoint_raises_error(self):
        """Test that single datapoint raises ValueError."""
        single_point = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            bottleneck(single_point)

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = bottleneck(data)
        assert isinstance(result, float)

    def test_three_points(self):
        # cosine distances
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = bottleneck(data)

        # dot product between [0, 1] (or [1,0])) and [0.5, 0.5]
        # 0 * 0.5 + 1 * 0.5 = 0.5
        # cosine similarity (divide dot product by vector norms):
        # 0.5/(1 * sqrt(0.5^2 + 0.5^2))
        assert np.isclose(result, 1 - 0.5/np.sqrt(0.5))

        # vectors are all pointing in the same direction,
        # so cosine sim = 1 --> distance = 0
        data = [[1, 0], [3, 0], [5, 0]]
        result = bottleneck(data)
        assert result == 0.0

        # duplicates
        data = [[-1, 0], [-1, 0], [-1, 0]]
        result = bottleneck(data)
        assert result == 0.0


class TestEnergy:

    def test_duplicates(self):
        data = [[0, 1], [0, 1]]
        # - 1/0.1 = -10
        assert np.isclose(energy(data, epsilon=0.1), -10)

        # 4 / (sqrt(2) * sqrt(8)) = 1, so distance is 0
        data = [[1, 1], [2, 2]]
        # - 1/0.1 = -10
        assert np.isclose(energy(data, epsilon=0.1), -10)


    def test_three_datapoints(self):
        # cosine distances
        data = [[0, 1], [1, 0], [1, 1]]

        # Pairwise cosine distances:
        # [0, 1] and [1, 0]: orthogonal, cosine similarity = 0, distance = 1.0
        # [0, 1] and [1, 1]: 1 - (1 / (sqrt(0^2 + 1^2) * sqrt(1^2 + 1^2))) ~ 0.2929
        # [1, 0] vs [1, 1]: 1 - (1 / (sqrt(1^2 + 0^2) * sqrt(1^2 + 1^2))) ~ 0.2929
        # So, distances are [1.0, 0.2929, 0.2929]
        # Energy = -(1/3) * (1/1 + 1/0.2929 + 1/0.2929)


        assert np.isclose(energy(data), -2.6095)


class TestDistanceDispersion:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute distances for empty data"):
            distance_dispersion([])

    def test_single_datapoint_raises_error(self):
        """Test that single datapoint raises ValueError."""
        single_point = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Cannot compute distances for single data point"):
            distance_dispersion(single_point)

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = distance_dispersion(data)
        assert isinstance(result, float)

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        sum_dist = distance_dispersion(data, metric="euclidean")
        expected = np.sqrt(2)  # sqrt((1-0)^2 + (1-0)^2)
        assert np.isclose(sum_dist, expected)

    def test_three_points_euclidean(self):
        """Test with three points using euclidean distance."""
        data = [[0, 0], [1, 0], [0, 1]]
        # Manual calculation: dist(p0,p1) = 1, dist(p0,p2) = 1, dist(p1,p2) = sqrt(2)
        expected_distances = [1.0, 1.0, np.sqrt(2)]
        expected_sum = np.sum(expected_distances)

        sum_dist = distance_dispersion(data, metric="euclidean")
        assert np.isclose(sum_dist, expected_sum)

    def test_relationship_with_mean(self):
        """Test the mathematical relationship between dispersion and mean."""
        np.random.seed(123)
        data = np.random.rand(4, 2)  # 4 points = 6 pairwise distances

        mean_dist = mean_pairwise_distance(data)
        sum_dist = distance_dispersion(data)

        # sum = mean * number_of_pairs
        n = len(data)
        num_pairs = n * (n - 1) // 2  # C(n,2)

        expected_sum = mean_dist * num_pairs
        assert np.isclose(sum_dist, expected_sum)




class TestConvexHullVolume:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute convex hull for empty data"):
            convex_hull_volume([])

    def test_few_points_raises_error(self):
        """Test behavior when there are insufficient points for the dimension."""
        with pytest.raises(ValueError, match="Cannot compute convex hull for fewer than dimension\+1 {} points \(got {}\)".format(4, 2)):
            convex_hull_volume([[0, 0, 0], [1, 1, 1]])

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 0], [1, 0], [0, 1]]
        result = convex_hull_volume(data)
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_collinear_points_zero_volume(self):
        """Test that collinear points have zero area."""
        line = [[0, 0], [1, 0], [2, 0]]
        assert convex_hull_volume(line) == 0.0

    def test_known_geometric_shapes(self):
        """Test convex hull volume for known geometric shapes."""
        # Triangle with known area
        triangle = [[0, 0], [1, 0], [0, 1]]
        area = convex_hull_volume(triangle)
        expected_area = 0.5  # Area of right triangle with legs 1,1
        assert np.isclose(area, expected_area)

        # Square with known area
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        area = convex_hull_volume(square)
        expected_area = 1.0  # Unit square
        assert np.isclose(area, expected_area)

        # 3D tetrahedron
        tetrahedron = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        volume = convex_hull_volume(tetrahedron)
        expected_volume = 1 / 6  # Volume of unit tetrahedron
        assert np.isclose(volume, expected_volume, rtol=1e-10)



class TestClusterInertiaDiversity:

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute cluster inertia for empty data"):
            cluster_inertia_diversity([])

    def test_single_datapoint_returns_zero(self):
        """Test that single datapoint raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute cluster centers and thus inertia for fewer than 2 datapoints"):
            cluster_inertia_diversity([[1, 2, 3]])

    def test_two_points_positive_inertia(self):
        """Test that two points produce positive inertia."""
        data = [[0, 0], [1, 1]]
        inertia = cluster_inertia_diversity(data, n_clusters=1)
        assert inertia > 0

    def test_return_type(self):
        """Test that function returns Python float."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = cluster_inertia_diversity(data)
        assert isinstance(result, float)

    def test_spread_vs_clustered_points(self):
        """Test that spread out points have higher inertia than clustered points."""
        spread_points = [[0, 0], [10, 0], [0, 10], [10, 10]]
        clustered_points = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]]

        spread_inertia = cluster_inertia_diversity(spread_points, n_clusters=2)
        clustered_inertia = cluster_inertia_diversity(clustered_points, n_clusters=2)

        assert spread_inertia > clustered_inertia

    def test_single_cluster_equals_centroid_distance(self):
        """Test that single cluster (k=1) equals distance to overall centroid."""
        data = [[1, 1], [2, 2], [3, 3]]
        inertia_k1 = cluster_inertia_diversity(data, n_clusters=1)

        # Manual calculation: centroid = [2, 2], inertia = sum of squared distances
        centroid = np.array([2, 2])
        manual_inertia = sum(np.sum((np.array(point) - centroid) ** 2) for point in data)

        assert np.isclose(inertia_k1, manual_inertia)

    def test_cluster_count_adjustment(self):
        """Test that function handles more clusters than points."""
        few_points = [[0, 0], [1, 1]]
        inertia = cluster_inertia_diversity(few_points, n_clusters=10)  # Should auto-adjust to 1
        assert inertia > 0

    def test_reproducibility(self):
        """Test that results are reproducible due to random_state."""
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]

        result1 = cluster_inertia_diversity(data, n_clusters=2)
        result2 = cluster_inertia_diversity(data, n_clusters=2)

        assert result1 == result2  # Should be exactly equal due to random_state




