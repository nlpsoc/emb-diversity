from measure_diversity import calculate_diversity

def test_calculate_diversity_basic():
    data = [[1, 0], [1, 1], [1, 0], [0, 1]]
    # Most common row is [1, 0] (twice out of 4)
    assert calculate_diversity(data) == 0.5

def test_all_same():
    data = [[0, 0]] * 5
    assert calculate_diversity(data) == 0.0

def test_all_unique():
    data = [[i, i] for i in range(4)]
    assert calculate_diversity(data) == 1 - (1 / 4)

def test_empty():
    assert calculate_diversity([]) == 0.0