from emb_diversity import dist_dispersion, mean_pw_dist, cluster_inertia, \
    convex_hull_volume_2d, energy, graph_entropy, diameter, sum_diameter, bottleneck, sum_bottleneck, hamdiv, log_determinant, dcscore, bins_entropy, renyi_entropy
import pytest
import numpy as np


class TestMeanPairwiseDistance:

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        mean_dist = mean_pw_dist(data, metric="euclidean")["value"]
        expected = np.sqrt(2)  # sqrt((1-0)^2 + (1-0)^2)
        assert np.isclose(mean_dist, expected)

    def test_three_points_euclidean(self):
        """Test with three points using euclidean distance."""
        data = [[0, 0], [1, 0], [0, 1]]
        # Manual calculation: dist(p0,p1) = 1, dist(p0,p2) = 1, dist(p1,p2) = sqrt(2)
        expected_distances = [1.0, 1.0, np.sqrt(2)]
        expected_mean = np.mean(expected_distances)

        mean_dist = mean_pw_dist(data, metric="euclidean")["value"]
        assert np.isclose(mean_dist, expected_mean)

    def test_different_metrics(self):
        """Test that different metrics produce different results."""
        data = [[1, 2], [3, 4], [5, 6]]

        euclidean_mean = mean_pw_dist(data, metric="euclidean")["value"]
        cosine_mean = mean_pw_dist(data, metric="cosine")["value"]
        manhattan_mean = mean_pw_dist(data, metric="cityblock")["value"]  # cityblock is Manhattan distance

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

        custom_mean = mean_pw_dist(data, metric=custom_metric)["value"]
        manhattan_mean = mean_pw_dist(data, metric="cityblock")["value"]

        # Should be the same since our custom metric is Manhattan distance
        assert np.isclose(custom_mean, manhattan_mean)

    def test_known_mathematical_values(self):
        """Test specific cases with known expected values."""
        # Two orthogonal unit vectors in cosine space
        assert np.isclose(mean_pw_dist([[1, 0], [0, 1]], metric="cosine")["value"], 1.0)

        # Three points: two identical, one different - Distances: (1,1,0) -> mean = 2/3
        assert np.isclose(mean_pw_dist([[1, 0], [1, 0], [0, 1]], metric="cosine")["value"], 2 / 3)

        # Scale invariance for cosine distance
        small_vectors = [[0.5, 0.5], [-0.5, -0.5]]
        large_vectors = [[1, 1], [-1, -1]]
        assert np.isclose(
            mean_pw_dist(small_vectors, metric="cosine")["value"],
            mean_pw_dist(large_vectors, metric="cosine")["value"]
        )


class TestDiameter:

    def test_three_points(self):
        # cosine distances

        # cosine distance between the first two points is 1 (as they are orthogonal)
        # so cosine similarity is 0 --> distance = 1
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = diameter(data)["value"]
        assert result == 1.0

        # vectors are all pointing in the same direction,
        # so cosine sim = 1 --> distance = 0
        data = [[1, 0], [3, 0], [5, 0]]
        result = diameter(data)["value"]
        assert result == 0.0

        # duplicates
        data = [[-1, 0], [-1, 0], [-1, 0]]
        result = diameter(data)["value"]
        assert result == 0.0



class TestSumDiameter:

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        expected = 2 * np.sqrt(2)  # Both points have max distance sqrt(2) to the other
        assert np.isclose(sum_diam, expected)

    def test_three_points_symmetric(self):
        """Test with three symmetric points (equilateral triangle)."""
        # Three points forming equilateral triangle with unit distances
        data = [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        # Each point has max distance 1.0 to one of the other points
        expected = 3.0  # 1.0 + 1.0 + 1.0
        assert np.isclose(sum_diam, expected)

    def test_collinear_points(self):
        """Test with collinear points."""
        # Three collinear points: [0,0], [1,0], [2,0]
        data = [[0, 0], [1, 0], [2, 0]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        # Max distances: [0]->2.0, [1]->2.0 (either direction), [2]->2.0
        # Actually: [0] max is 2.0 (to [2]), [1] max is 1.0 (to [0] or [2]), [2] max is 2.0 (to [0])
        # sum = 2.0 + 1.0 + 2.0 = 5.0
        expected = 5.0
        assert np.isclose(sum_diam, expected)

    def test_identical_points(self):
        """Test with identical points - all distances are zero."""
        data = [[1, 1], [1, 1], [1, 1]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        assert sum_diam == 0.0

    def test_different_metrics(self):
        """Test that different metrics produce different results."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        
        euclidean_sum = sum_diameter(data, metric="euclidean")["value"]
        cosine_sum = sum_diameter(data, metric="cosine")["value"]
        
        # Different metrics should produce different results
        assert not np.isclose(euclidean_sum, cosine_sum)

    def test_normalize_by_n(self):
        """Test the normalize_by_n parameter."""
        data = [[0, 0], [1, 1], [2, 2]]
        sum_diam = sum_diameter(data, metric="euclidean", normalize_by_n=False)["value"]
        avg_diam = sum_diameter(data, metric="euclidean", normalize_by_n=True)["value"]
        
        n = len(data)
        assert np.isclose(avg_diam, sum_diam / n)

    def test_orthogonal_vectors_cosine(self):
        """Test with orthogonal vectors using cosine distance."""
        # Two orthogonal unit vectors
        data = [[1, 0], [0, 1]]
        sum_diam = sum_diameter(data, metric="cosine")["value"]
        # Cosine distance between orthogonal unit vectors is 1.0
        # Each point has max distance 1.0 to the other
        expected = 2.0
        assert np.isclose(sum_diam, expected)

    def test_relationship_with_diameter(self):
        """Test mathematical relationship: sum_diameter >= diameter."""
        data = [[0, 0], [1, 1], [0, 1], [1, 0]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        max_diam = diameter(data, metric="euclidean")["value"]
        
        # sum_diameter should be >= diameter (it sums multiple max distances)
        assert sum_diam >= max_diam

    def test_scale_invariance_normalized_cosine(self):
        """Test that scaling doesn't affect cosine-based sum_diameter."""
        small = [[0.5, 0.5], [-0.5, -0.5]]
        large = [[1.0, 1.0], [-1.0, -1.0]]
        
        sum_small = sum_diameter(small, metric="cosine")["value"]
        sum_large = sum_diameter(large, metric="cosine")["value"]
        
        # Cosine distance is scale-invariant
        assert np.isclose(sum_small, sum_large)

    def test_four_points_square(self):
        """Test with four points forming a square."""
        data = [[0, 0], [1, 0], [1, 1], [0, 1]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        
        # Max distances from each corner:
        # [0,0]: sqrt(2) (diagonal to [1,1])
        # [1,0]: sqrt(2) (diagonal to [0,1])
        # [1,1]: sqrt(2) (diagonal to [0,0])
        # [0,1]: sqrt(2) (diagonal to [1,0])
        expected = 4 * np.sqrt(2)
        assert np.isclose(sum_diam, expected)

    def test_two_clusters_distant_points(self):
        """Test with two well-separated clusters."""
        # Cluster 1 near origin, Cluster 2 far away
        data = [[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]
        
        # Points in cluster 1 have max distance to cluster 2 (~14.14)
        # Points in cluster 2 have max distance to cluster 1 (~14.14)
        # sum ~ 4 * 14.14 ≈ 56.6
        assert sum_diam > 50

    def test_custom_metric_function(self):
        """Test using a custom metric function."""
        def manhattan_distance(u, v):
            return np.sum(np.abs(u - v))
        
        data = [[0, 0], [1, 1], [2, 2]]
        
        custom_sum = sum_diameter(data, metric=manhattan_distance)["value"]
        cityblock_sum = sum_diameter(data, metric="cityblock")["value"]
        
        assert np.isclose(custom_sum, cityblock_sum)

        
class TestBottleneck:

    def test_three_points(self):
        # cosine distances
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = bottleneck(data)["value"]

        # dot product between [0, 1] (or [1,0])) and [0.5, 0.5]
        # 0 * 0.5 + 1 * 0.5 = 0.5
        # cosine similarity (divide dot product by vector norms):
        # 0.5/(1 * sqrt(0.5^2 + 0.5^2))
        assert np.isclose(result, 1 - 0.5/np.sqrt(0.5))

        # vectors are all pointing in the same direction,
        # so cosine sim = 1 --> distance = 0
        data = [[1, 0], [3, 0], [5, 0]]
        result = bottleneck(data)["value"]
        assert result == 0.0

        # duplicates
        data = [[-1, 0], [-1, 0], [-1, 0]]
        result = bottleneck(data)["value"]
        assert result == 0.0


class TestSumBottleneck:

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        # Both points have min distance sqrt(2) to the (only) other point
        expected = 2 * np.sqrt(2)
        assert np.isclose(sum_bn, expected)

    def test_three_points_equilateral(self):
        """Test with three points forming an equilateral triangle."""
        # All pairwise distances equal 1 -> each point's nearest = 1
        data = [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        expected = 3.0  # 1.0 + 1.0 + 1.0
        assert np.isclose(sum_bn, expected)

    def test_collinear_points(self):
        """Test with collinear points."""
        # Points [0,0], [1,0], [2,0]
        # Nearest distances: [0]->1.0, [1]->1.0, [2]->1.0
        data = [[0, 0], [1, 0], [2, 0]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        expected = 3.0
        assert np.isclose(sum_bn, expected)

    def test_identical_points(self):
        """Test with identical points - all distances are zero."""
        data = [[1, 1], [1, 1], [1, 1]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        assert sum_bn == 0.0

    def test_different_metrics(self):
        """Test that different metrics produce different results."""
        data = [[0, 1], [1, 0], [0.5, 0.5]]

        euclidean_sum = sum_bottleneck(data, metric="euclidean")["value"]
        cosine_sum = sum_bottleneck(data, metric="cosine")["value"]

        assert not np.isclose(euclidean_sum, cosine_sum)

    def test_normalize_by_n(self):
        """Test the normalize_by_n parameter."""
        data = [[0, 0], [1, 1], [2, 2]]
        sum_bn = sum_bottleneck(data, metric="euclidean", normalize_by_n=False)["value"]
        avg_bn = sum_bottleneck(data, metric="euclidean", normalize_by_n=True)["value"]

        n = len(data)
        assert np.isclose(avg_bn, sum_bn / n)

    def test_orthogonal_vectors_cosine(self):
        """Test with orthogonal vectors using cosine distance."""
        # Two orthogonal unit vectors: cosine distance = 1.0
        data = [[1, 0], [0, 1]]
        sum_bn = sum_bottleneck(data, metric="cosine")["value"]
        expected = 2.0  # each row's min = 1.0
        assert np.isclose(sum_bn, expected)

    def test_relationship_with_bottleneck(self):
        """sum_bottleneck >= n * bottleneck (each row's min >= global min)."""
        data = [[0, 0], [1, 1], [0, 1], [1, 0]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        min_bn = bottleneck(data, metric="euclidean")["value"]
        n = len(data)

        assert sum_bn >= n * min_bn - 1e-12

    def test_relationship_with_sum_diameter(self):
        """sum_bottleneck <= sum_diameter (per-row min <= per-row max)."""
        data = [[0, 0], [1, 1], [0, 1], [1, 0]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        sum_diam = sum_diameter(data, metric="euclidean")["value"]

        assert sum_bn <= sum_diam + 1e-12

    def test_scale_invariance_cosine(self):
        """Cosine-based sum_bottleneck should be scale-invariant."""
        small = [[0.5, 0.5], [-0.5, -0.5]]
        large = [[1.0, 1.0], [-1.0, -1.0]]

        sum_small = sum_bottleneck(small, metric="cosine")["value"]
        sum_large = sum_bottleneck(large, metric="cosine")["value"]

        assert np.isclose(sum_small, sum_large)

    def test_four_points_square(self):
        """Test with four points forming a unit square."""
        # Each corner's nearest neighbour is at distance 1 (adjacent corner)
        data = [[0, 0], [1, 0], [1, 1], [0, 1]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        expected = 4.0  # 1 + 1 + 1 + 1
        assert np.isclose(sum_bn, expected)

    def test_two_tight_clusters(self):
        """With tight clusters, sum_bottleneck stays small even when clusters are far."""
        data = [[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]]
        sum_bn = sum_bottleneck(data, metric="euclidean")["value"]
        # Each point's nearest is its cluster-mate at distance sqrt(0.02)
        expected = 4.0 * np.sqrt(0.02)
        assert np.isclose(sum_bn, expected)

    def test_custom_metric_function(self):
        """Test using a custom metric function."""
        def manhattan_distance(u, v):
            return np.sum(np.abs(u - v))

        data = [[0, 0], [1, 1], [2, 2]]

        custom_sum = sum_bottleneck(data, metric=manhattan_distance)["value"]
        cityblock_sum = sum_bottleneck(data, metric="cityblock")["value"]

        assert np.isclose(custom_sum, cityblock_sum)


class TestEnergy:

    def test_duplicates(self):
        data = [[0, 1], [0, 1]]
        # - 1/0.1 = -10
        assert np.isclose(energy(data, epsilon=0.1)["value"], -10)

        # 4 / (sqrt(2) * sqrt(8)) = 1, so distance is 0
        data = [[1, 1], [2, 2]]
        # - 1/0.1 = -10
        assert np.isclose(energy(data, epsilon=0.1)["value"], -10)


    def test_three_datapoints(self):
        # cosine distances
        data = [[0, 1], [1, 0], [1, 1]]

        # Pairwise cosine distances:
        # [0, 1] and [1, 0]: orthogonal, cosine similarity = 0, distance = 1.0
        # [0, 1] and [1, 1]: 1 - (1 / (sqrt(0^2 + 1^2) * sqrt(1^2 + 1^2))) ~ 0.2929
        # [1, 0] vs [1, 1]: 1 - (1 / (sqrt(1^2 + 0^2) * sqrt(1^2 + 1^2))) ~ 0.2929
        # So, distances are [1.0, 0.2929, 0.2929]
        # Energy = -(1/3) * (1/1 + 1/0.2929 + 1/0.2929)

        assert np.isclose(energy(data)["value"], -2.6095)


class TestDistanceDispersion:

    def test_two_points_basic(self):
        """Test basic functionality with two points."""
        data = [[0, 0], [1, 1]]
        sum_dist = dist_dispersion(data, metric="euclidean")["value"]
        expected = np.sqrt(2)  # sqrt((1-0)^2 + (1-0)^2)
        assert np.isclose(sum_dist, expected)

    def test_three_points_euclidean(self):
        """Test with three points using euclidean distance."""
        data = [[0, 0], [1, 0], [0, 1]]
        # Manual calculation: dist(p0,p1) = 1, dist(p0,p2) = 1, dist(p1,p2) = sqrt(2)
        expected_distances = [1.0, 1.0, np.sqrt(2)]
        expected_sum = np.sum(expected_distances)

        sum_dist = dist_dispersion(data, metric="euclidean")["value"]
        assert np.isclose(sum_dist, expected_sum)

    def test_relationship_with_mean(self):
        """Test the mathematical relationship between dispersion and mean."""
        np.random.seed(123)
        data = np.random.rand(4, 2)  # 4 points = 6 pairwise distances

        mean_dist = mean_pw_dist(data)["value"]
        sum_dist = dist_dispersion(data)["value"]

        # sum = mean * number_of_pairs
        n = len(data)
        num_pairs = n * (n - 1) // 2  # C(n,2)

        expected_sum = mean_dist * num_pairs
        assert np.isclose(sum_dist, expected_sum)




class TestConvexHullVolume2D:

    def test_few_points_raises_error(self):
        """Test behavior when there are fewer than 3 points (cannot form 2D hull)."""
        with pytest.raises(ValueError, match=r"Cannot compute 2D convex hull for fewer than 3 points \(got 2\)"):
            convex_hull_volume_2d([[0, 0], [1, 1]])["value"]

    def test_collinear_points_zero_volume(self):
        """Test that collinear points have zero area."""
        # Input is already 2D, so reduction is a no-op and points stay collinear.
        line = [[0, 0], [1, 0], [2, 0]]
        assert convex_hull_volume_2d(line)["value"] == 0.0

    def test_known_2d_shapes(self):
        """For inputs already in 2D, reduction is a no-op and areas are exact."""
        # Triangle with known area
        triangle = [[0, 0], [1, 0], [0, 1]]
        area = convex_hull_volume_2d(triangle)["value"]
        expected_area = 0.5  # Area of right triangle with legs 1,1
        assert np.isclose(area, expected_area)

        # Square with known area
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        area = convex_hull_volume_2d(square)["value"]
        expected_area = 1.0  # Unit square
        assert np.isclose(area, expected_area)

    def test_numpy_array_input(self):
        """Accept a numpy array without raising the ambiguous-truth-value ValueError."""
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        area = convex_hull_volume_2d(triangle)["value"]
        assert np.isclose(area, 0.5)

    def test_high_dim_input_uses_umap_projection(self):
        """For >2D input, UMAP projection is invoked and yields a positive area."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(50, 10))
        area = convex_hull_volume_2d(data, random_state=42)["value"]
        assert isinstance(area, float)
        assert area > 0
        assert np.isfinite(area)

    def test_umap_failure_falls_back_to_first_two_columns(self, monkeypatch):
        """If umap-learn cannot be imported, fall back to the first 2 columns with a warning."""
        import sys
        # Force `import umap` to raise ImportError inside _reduce_to_2d.
        monkeypatch.setitem(sys.modules, "umap", None)
        # First 2 columns form a known triangle; the third column is ignored.
        data = [[0.0, 0.0, 99.0], [1.0, 0.0, 99.0], [0.0, 1.0, 99.0]]
        with pytest.warns(UserWarning, match="UMAP reduction to 2D failed"):
            area = convex_hull_volume_2d(data)["value"]
        assert np.isclose(area, 0.5)



class TestClusterInertiaDiversity:

    def test_single_datapoint_raises(self):
        """Test that single datapoint raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            cluster_inertia([[1, 2, 3]])["value"]

    def test_two_points_positive_inertia(self):
        """Test that two points produce positive inertia."""
        data = [[0, 0], [1, 1]]
        inertia = cluster_inertia(data, n_clusters=1)["value"]
        assert inertia > 0

    def test_spread_vs_clustered_points(self):
        """Test that spread out points have higher inertia than clustered points."""
        spread_points = [[0, 0], [10, 0], [0, 10], [10, 10]]
        clustered_points = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]]

        spread_inertia = cluster_inertia(spread_points, n_clusters=2)["value"]
        clustered_inertia = cluster_inertia(clustered_points, n_clusters=2)["value"]

        assert spread_inertia > clustered_inertia

    def test_single_cluster_equals_centroid_distance(self):
        """Test that single cluster (k=1) equals distance to overall centroid."""
        data = [[1, 1], [2, 2], [3, 3]]
        inertia_k1 = cluster_inertia(data, n_clusters=1)["value"]

        # Manual calculation: centroid = [2, 2], inertia = sum of squared distances
        centroid = np.array([2, 2])
        manual_inertia = sum(np.sum((np.array(point) - centroid) ** 2) for point in data)

        assert np.isclose(inertia_k1, manual_inertia)

    def test_cluster_count_adjustment(self):
        """Test that function handles more clusters than points."""
        few_points = [[0, 0], [1, 1]]
        inertia = cluster_inertia(few_points, n_clusters=10)["value"]  # Should auto-adjust to 1
        assert inertia > 0

    def test_reproducibility(self):
        """Test that results are reproducible due to random_state."""
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]

        result1 = cluster_inertia(data, n_clusters=2)["value"]
        result2 = cluster_inertia(data, n_clusters=2)["value"]

        assert result1 == result2  # Should be exactly equal due to random_state



class TestGraphEntropy:

    def test_identical_points_zero_entropy(self):
        """Test that identical points result in zero entropy."""
        data = np.array([[1, 1], [1, 1], [1, 1]])
        result = graph_entropy(data, metric="euclidean")["value"]
        assert np.isclose(result, 0.0)

    def test_orthogonal_vectors_known_entropy(self):
        """Test entropy calculation for known orthogonal vectors."""
        data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        result = graph_entropy(data, metric="cosine")["value"]
        expected_local_entropy = np.log(2)
        expected_total_entropy = 3 * expected_local_entropy
        
        assert np.isclose(result, expected_total_entropy)

    def test_scale_invariance_cosine(self):
        """Test that scaling vectors does not change cosine-based graph entropy."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaled_data = np.array([[10, 20], [30, 40], [50, 60]])

        entropy_original = graph_entropy(data, metric="cosine")["value"]
        entropy_scaled = graph_entropy(scaled_data, metric="cosine")["value"]

        assert np.isclose(entropy_original, entropy_scaled)

    def test_different_metrics(self):
        """Test that different metrics produce different entropy values."""
        data = np.array([[0, 0], [1, 1], [2, 2]])
        
        euclidean_ent = graph_entropy(data, metric="euclidean")["value"]
        cosine_ent = graph_entropy(data, metric="cosine")["value"]

        assert not np.isclose(euclidean_ent, cosine_ent)



class TestDCScore:

    def test_two_orthogonal_vectors_cs_kernel(self):
        """
        For two orthogonal unit vectors with cosine-like kernel:
        X = [[1,0], [0,1]], normalize=True, tau=1

        K = [[1, 0],
             [0, 1]]

        Row-wise softmax:
          row1: [e^1 / (e^1 + e^0), e^0 / (e^1 + e^0)]
          row2: [e^0 / (e^1 + e^0), e^1 / (e^1 + e^0)]

        So DCScore = sum diag(P) = 2 * e^1 / (e^1 + e^0)
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        score = dcscore(data, kernel_type="cs", tau=1.0, normalize=True)["value"]

        e1 = np.exp(1.0)
        expected = 2.0 * e1 / (e1 + 1.0)
        assert np.isclose(score, expected)

    def test_scale_invariance_with_normalize(self):
        """
        With normalize=True, scaling the vectors should not change the score
        for the cosine-like kernel.
        """
        small = [[0.5, 0.5], [-0.5, -0.5]]
        large = [[1.0, 1.0], [-1.0, -1.0]]

        score_small = dcscore(small, kernel_type="cs", tau=1.0, normalize=True)["value"]
        score_large = dcscore(large, kernel_type="cs", tau=1.0, normalize=True)["value"]

        assert np.isclose(score_small, score_large)

    def test_higher_tau_flattens_softmax(self):
        """
        Larger tau should make the softmax more uniform, which typically
        reduces the diagonal dominance and thus the DCScore.

        We don't assert a strict inequality for all possible datasets,
        but we can use a simple asymmetric example where we expect
        smaller DCScore for larger tau.
        """
        data = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]

        score_tau_small = dcscore(data, kernel_type="cs", tau=0.5, normalize=True)["value"]
        score_tau_large = dcscore(data, kernel_type="cs", tau=5.0, normalize=True)["value"]

        assert score_tau_small > score_tau_large


class TestLogDeterminantDiversity:

    def test_negative_eps_raises_error(self):
        """Test that negative eps raises ValueError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="eps must be positive"):
            log_determinant(data, eps=-1.0)["value"]

    def test_zero_eps_raises_error(self):
        """Test that zero eps raises ValueError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="eps must be positive"):
            log_determinant(data, eps=0.0)["value"]

    def test_negative_tau_raises_error(self):
        """Test that negative tau raises ValueError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="tau must be positive"):
            log_determinant(data, tau=-1.0)["value"]

    def test_zero_tau_raises_error(self):
        """Test that zero tau raises ValueError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="tau must be positive"):
            log_determinant(data, tau=0.0)["value"]

    def test_unknown_kernel_type_raises_error(self):
        """Test that unknown kernel_type raises NotImplementedError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(NotImplementedError, match="Unknown kernel_type"):
            log_determinant(data, kernel_type="unknown")["value"]

    def test_poly_kernel_non_integer_tau_raises_error(self):
        """Test that poly kernel with non-integer tau raises ValueError."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="For 'poly' kernel, tau must be an integer"):
            log_determinant(data, kernel_type="poly", tau=2.5)["value"]

    def test_two_orthogonal_vectors_cs_kernel(self):
        """
        Test with two orthogonal unit vectors using cosine-like kernel.
        X = [[1,0], [0,1]], normalize=True, tau=1
        
        K = [[1, 0],
             [0, 1]]
        
        A = K + eps*I = [[1+eps, 0],
                         [0, 1+eps]]
        
        logdet = log((1+eps)^2) = 2*log(1+eps)
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        eps = 1e-6
        result = log_determinant(data, kernel_type="cs", tau=1.0, normalize=True, eps=eps)["value"]
        expected = 2.0 * np.log(1.0 + eps)
        assert np.isclose(result, expected, rtol=1e-5)

    def test_identical_vectors_cs_kernel(self):
        """
        Test with identical vectors. The similarity matrix should have
        high diagonal values and off-diagonal values equal to 1 (for normalized vectors).
        For identical normalized vectors, K = [[1, 1], [1, 1]], which is singular.
        With eps*I added, logdet should be finite but can be negative and small.
        """
        data = [[1.0, 0.0], [1.0, 0.0]]
        result = log_determinant(data, kernel_type="cs", normalize=True, eps=1e-6)["value"]
        # For identical vectors, the matrix is near-singular, so logdet can be negative
        # With eps=1e-6, the matrix becomes [[1+eps, 1], [1, 1+eps]]
        # det = (1+eps)^2 - 1 = 2*eps + eps^2, so logdet = log(2*eps + eps^2) ≈ log(2e-6) ≈ -13.1
        assert result < 0  # Should be negative for near-singular matrix
        assert np.isfinite(result)  # Should be finite
        assert isinstance(result, float)

    def test_different_kernel_types_produce_different_results(self):
        """Test that different kernel types produce different logdet values."""
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result_cs = log_determinant(data, kernel_type="cs", tau=1.0)["value"]
        result_rbf = log_determinant(data, kernel_type="rbf", tau=1.0)["value"]
        result_lap = log_determinant(data, kernel_type="lap", tau=1.0)["value"]
        
        # Results should be different for different kernels
        assert not np.isclose(result_cs, result_rbf)
        assert not np.isclose(result_cs, result_lap)
        assert not np.isclose(result_rbf, result_lap)

    def test_scale_invariance_with_normalize_cs(self):
        """
        With normalize=True, scaling the vectors should not change the logdet
        for the cosine-like kernel.
        """
        small = [[0.5, 0.5], [-0.5, -0.5]]
        large = [[1.0, 1.0], [-1.0, -1.0]]
        
        score_small = log_determinant(small, kernel_type="cs", normalize=True, tau=1.0)["value"]
        score_large = log_determinant(large, kernel_type="cs", normalize=True, tau=1.0)["value"]
        
        assert np.isclose(score_small, score_large, rtol=1e-5)

    def test_tau_affects_result(self):
        """
        Different tau values should produce different logdet values.
        For cs kernel, larger tau scales down the similarity matrix.
        """
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result_tau_small = log_determinant(data, kernel_type="cs", tau=0.5, normalize=True)["value"]
        result_tau_large = log_determinant(data, kernel_type="cs", tau=5.0, normalize=True)["value"]
        
        # Results should be different
        assert not np.isclose(result_tau_small, result_tau_large)

    def test_eps_affects_result(self):
        """
        Different eps values should produce different logdet values.
        Larger eps makes the matrix more positive definite.
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        
        result_eps_small = log_determinant(data, kernel_type="cs", eps=1e-8)["value"]
        result_eps_large = log_determinant(data, kernel_type="cs", eps=1e-4)["value"]
        
        # Results should be different
        assert not np.isclose(result_eps_small, result_eps_large)

    def test_cholesky_vs_slogdet_consistency(self):
        """
        Test that Cholesky and slogdet methods produce consistent results
        when both are applicable.
        """
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result_cholesky = log_determinant(data, kernel_type="cs", use_cholesky=True)["value"]
        result_slogdet = log_determinant(data, kernel_type="cs", use_cholesky=False)["value"]
        
        # Should be very close (within numerical precision)
        assert np.isclose(result_cholesky, result_slogdet, rtol=1e-5)

    def test_poly_kernel_with_integer_tau(self):
        """Test polynomial kernel with valid integer tau."""
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result = log_determinant(data, kernel_type="poly", tau=2, eps=1e-6)["value"]
        
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_rbf_kernel_basic(self):
        """Test RBF kernel produces valid results."""
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result = log_determinant(data, kernel_type="rbf", tau=1.0, eps=1e-6)["value"]
        
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_lap_kernel_basic(self):
        """Test Laplacian kernel produces valid results."""
        data = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        
        result = log_determinant(data, kernel_type="lap", tau=1.0, eps=1e-6)["value"]
        
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_diversity_increases_with_more_diverse_data(self):
        """
        Test that more diverse (spread out) data produces higher logdet values
        than less diverse (clustered) data.
        """
        # More diverse: orthogonal vectors
        diverse_data = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        
        # Less diverse: similar vectors
        clustered_data = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]
        
        diverse_ldd = log_determinant(diverse_data, kernel_type="cs", normalize=True)["value"]
        clustered_ldd = log_determinant(clustered_data, kernel_type="cs", normalize=True)["value"]
        
        # More diverse data should have higher logdet (more volume in feature space)
        assert diverse_ldd > clustered_ldd


class TestBinsBasedEntropyPCA:
    def test_invalid_shape_raises_error(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            bins_entropy([1, 2, 3])["value"]  # 1D input

    def test_invalid_bins_raises_error(self):
        data = [[0, 1], [1, 0], [0.5, 0.5]]

        with pytest.raises(ValueError, match="must be positive integers"):
            bins_entropy(data, n_bins_x=0, n_bins_y=5)["value"]
        with pytest.raises(ValueError, match="must be positive integers"):
            bins_entropy(data, n_bins_x=5, n_bins_y=-1)["value"]

    def test_return_type_is_python_float(self):
        data = [[0, 1], [1, 0], [0.5, 0.5]]
        result = bins_entropy(data, projection="pca")["value"]
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_normalized_entropy_in_range(self):
        np.random.seed(42)
        data = np.random.randn(50, 8)
        entropy = bins_entropy(data, n_bins_x=5, n_bins_y=5, normalize=True, projection="pca")["value"]
        assert 0.0 <= entropy <= 1.0

    def test_unnormalized_vs_normalized(self):
        np.random.seed(42)
        data = np.random.randn(50, 8)

        normalized = bins_entropy(data, n_bins_x=5, n_bins_y=5, normalize=True, projection="pca")["value"]
        unnormalized = bins_entropy(data, n_bins_x=5, n_bins_y=5, normalize=False, projection="pca")["value"]

        # Unnormalized entropy should be >= normalized entropy (since normalization divides by log factor > 1)
        assert unnormalized >= normalized

    def test_deterministic_results(self):
        # PCA is deterministic, so results should match exactly (or extremely close numerically).
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0.5, 1.5, 2.5]], dtype=float)
        r1 = bins_entropy(data, n_bins_x=3, n_bins_y=3, normalize=True, projection="pca")["value"]
        r2 = bins_entropy(data, n_bins_x=3, n_bins_y=3, normalize=True, projection="pca")["value"]
        assert np.isclose(r1, r2, atol=1e-12)

    def test_all_points_in_same_bin_entropy_zero(self):
        # Near-identical points -> PCA projection collapses -> all in same bin -> entropy ~= 0
        data = np.array([[1,1,1],[1,1,1+1e-12],[1,1,1+2e-12]], dtype=float)


        entropy = bins_entropy(data, n_bins_x=2, n_bins_y=2, normalize=True, projection="pca")["value"]
        assert np.isclose(entropy, 0.0, atol=1e-12)

    def test_uniform_vs_clustered_relative(self):
        # Prefer relative comparisons (more stable than absolute thresholds)
        np.random.seed(0)

        # uniform in 2D
        uniform = np.random.uniform(-1, 1, size=(400, 2))

        # clustered in 2D
        clustered = np.random.normal(0, 0.05, size=(400, 2))

        e_uniform = bins_entropy(uniform, n_bins_x=6, n_bins_y=6, normalize=True, projection="pca")["value"]
        e_clustered = bins_entropy(clustered, n_bins_x=6, n_bins_y=6, normalize=True, projection="pca")["value"]
        assert e_uniform > e_clustered


    def test_non_square_bins(self):
        np.random.seed(42)
        data = np.random.randn(60, 6)

        e_5x10 = bins_entropy(data, n_bins_x=5, n_bins_y=10, normalize=True, projection="pca")["value"]
        e_10x5 = bins_entropy(data, n_bins_x=10, n_bins_y=5, normalize=True, projection="pca")["value"]
        assert isinstance(e_5x10, float)
        assert isinstance(e_10x5, float)
        assert 0.0 <= e_5x10 <= 1.0
        assert 0.0 <= e_10x5 <= 1.0

    def test_large_dataset_smoke(self):
        np.random.seed(42)
        data = np.random.randn(500, 32)
        entropy = bins_entropy(data, n_bins_x=10, n_bins_y=10, normalize=True, projection="pca")["value"]
        assert isinstance(entropy, float)
        assert 0.0 <= entropy <= 1.0
        assert entropy > 0.0

class TestRenyiKernelEntropy:

    def test_requires_at_least_2_datapoints(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            renyi_entropy([[1.0, 2.0, 3.0]])["value"]

    def test_return_type_is_python_float(self):
        data = [[1.0, 0.0], [0.0, 1.0]]
        score = renyi_entropy(data, alpha=2.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert isinstance(score, float)
        assert not isinstance(score, np.floating)

    def test_cs_orthogonal_vectors_alpha2_equals_log2(self):
        """
        X = [[1,0],[0,1]] with normalize=True, tau=1:
          K = I
          tr(K) = 2
          A = K / tr(K) = 0.5 I
          ||A||_F^2 = 0.5
          RKE_2 = -log(0.5) = log(2)
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        score = renyi_entropy(data, alpha=2.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert np.isclose(score, np.log(2.0), atol=1e-12)

    def test_cs_identical_vectors_alpha2_equals_zero(self):
        """
        Identical normalized vectors:
          K = [[1,1],[1,1]]
          tr(K)=2
          A=0.5*[[1,1],[1,1]]
          ||A||_F^2 = 1
          RKE_2 = -log(1)=0
        """
        data = [[1.0, 0.0], [1.0, 0.0]]
        score = renyi_entropy(data, alpha=2.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert np.isclose(score, 0.0, atol=1e-12)

    def test_cs_scale_invariance_when_normalize_true(self):
        """
        With normalize=True for cs, scaling vectors should not change K (up to numerical eps),
        hence RKE should be invariant.
        """
        small = [[0.5, 0.5], [-0.5, -0.5]]
        large = [[1.0, 1.0], [-1.0, -1.0]]

        s1 = renyi_entropy(small, alpha=2.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        s2 = renyi_entropy(large, alpha=2.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert np.isclose(s1, s2, atol=1e-12)

    def test_alpha1_von_neumann_entropy_orthogonal_equals_log2(self):
        """
        For orthogonal case above, eigenvalues of A are [0.5, 0.5],
        so von Neumann entropy = -sum p log p = log(2).
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        score = renyi_entropy(data, alpha=1.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert np.isclose(score, np.log(2.0), atol=1e-12)

    def test_general_alpha_uniform_eigs_equals_log2(self):
        """
        Same orthogonal setup => eigenvalues are uniform [0.5,0.5],
        so RKE_alpha should be log(2) for any alpha>0, alpha!=1 (and also for alpha=1).
        We'll test alpha=3.
        """
        data = [[1.0, 0.0], [0.0, 1.0]]
        score = renyi_entropy(data, alpha=3.0, kernel_type="cs", tau=1.0, normalize=True)["value"]
        assert np.isclose(score, np.log(2.0), atol=1e-12)

    def test_invalid_kernel_type_raises(self):
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(NotImplementedError, match="Unknown kernel_type"):
            renyi_entropy(data, kernel_type="wat")["value"]

    def test_invalid_tau_raises(self):
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="tau must be positive"):
            renyi_entropy(data, kernel_type="cs", tau=0.0)["value"]

    def test_invalid_alpha_raises(self):
        data = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_entropy(data, alpha=0.0)["value"]
