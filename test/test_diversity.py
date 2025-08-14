from measure_diversity import dummy_diversity, pairwise_diversity, two_d


def test_pairwise_diversity():
    assert pairwise_diversity( [[1, 0]]) == 0.0  # Single point, no pairs
    assert pairwise_diversity([[1, 0], [0, 1]]) == 1.0  # Two orthogonal points
    assert pairwise_diversity([[1, 0], [1, 0], [0, 1]]) == 2/3  # =(1+1+0)/3
    assert pairwise_diversity([[0.5, 0.5], [-0.5, -0.5]]) == pairwise_diversity([[1, 1], [-1, -1]])

    # test
    points_equi = two_d.create_normed_datapoints(n=8)
    points_first_quad = two_d.create_normed_datapoints(n=8, quads=[1])
    points_first_and_third_quad = two_d.create_normed_datapoints(n=8, quads=[1, 3])
    points_second_quad = two_d.create_normed_datapoints(n=8, quads=[2])

    assert pairwise_diversity(points_equi) > pairwise_diversity(points_first_and_third_quad)
    assert pairwise_diversity(points_equi) > pairwise_diversity(points_first_quad)
    assert pairwise_diversity(points_first_and_third_quad) > pairwise_diversity(points_first_quad)
    assert pairwise_diversity(points_first_quad) == pairwise_diversity(points_second_quad)


def test_calculate_diversity_basic():
    data = [[1, 0], [1, 1], [1, 0], [0, 1]]
    # Most common row is [1, 0] (twice out of 4)
    assert dummy_diversity(data) == 0.5

def test_all_same():
    data = [[0, 0]] * 5
    assert dummy_diversity(data) == 0.0

def test_all_unique():
    data = [[i, i] for i in range(4)]
    assert dummy_diversity(data) == 1 - (1 / 4)

def test_empty():
    assert dummy_diversity([]) == 0.0