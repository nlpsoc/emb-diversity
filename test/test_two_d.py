from measure_diversity.two_d import create_normed_datapoints, duplicate_dataset, create_toy_dataset1_axioms_challenges
import pytest
import numpy as np
from collections import Counter

class TestCreateNormedDatapoints:

    def test_empty_quadrants_error(self):
        """Test that empty quadrant list raises error."""
        with pytest.raises(ValueError, match="At least one quadrant must be specified"):
            create_normed_datapoints(quads=[])

    def test_custom_n_points(self):
        """Test creating different numbers of points."""
        for n in [1, 3, 5, 8, 10, 16]:
            points = create_normed_datapoints(n=n)
            assert len(points) == n

            # All points should be on unit circle
            for x, y in points:
                distance_from_origin = np.sqrt(x ** 2 + y ** 2)
                assert np.isclose(distance_from_origin, 1.0, rtol=1e-10)

    def test_single_quadrant(self):
        """Test points distributed in single quadrants."""
        # Test each quadrant individually
        for quad in [1, 2, 3, 4]:
            points = create_normed_datapoints(n=4, quads=[quad])
            assert len(points) == 4

            for x, y in points:
                # Check quadrant constraints
                if quad == 1:  # [0, π/2] -> x >= 0, y >= 0
                    assert x >= -1e-10 and y >= -1e-10
                elif quad == 2:  # [π/2, π] -> x <= 0, y >= 0
                    assert x <= 1e-10 and y >= -1e-10
                elif quad == 3:  # [π, 3π/2] -> x <= 0, y <= 0
                    assert x <= 1e-10 and y <= 1e-10
                elif quad == 4:  # [3π/2, 2π] -> x >= 0, y <= 0
                    assert x >= -1e-10 and y <= 1e-10

    def test_angular_distribution(self):
        """Test that points are roughly equidistantly distributed."""
        points = create_normed_datapoints(n=8)

        # Calculate angles from points
        angles = []
        for x, y in points:
            angle = np.arctan2(y, x)
            if angle < 0:
                angle += 2 * np.pi  # Convert to [0, 2π] range
            angles.append(angle)

        angles.sort()

        # Check that angles are roughly evenly spaced
        expected_spacing = 2 * np.pi / 8
        for i in range(len(angles)):
            next_angle = angles[(i + 1) % len(angles)]
            if i == len(angles) - 1:  # Wrap around case
                spacing = (2 * np.pi - angles[i]) + angles[0]
            else:
                spacing = next_angle - angles[i]

            # Allow some tolerance for equidistant spacing
            assert abs(spacing - expected_spacing) < 0.5


class TestDuplicateDatapoints:

    def test_duplicates_default(self):
        """Duplicate instances when no specific number of duplicates is provided."""
        dataset_orig = [[1,1], [0,0], [4.5, 3]]
        expected_result =  [[1,1], [1,1], [0,0], [0,0], [4.5, 3], [4.5, 3]]
        assert duplicate_dataset(dataset_orig) == expected_result

    def test_duplicates_num_specified(self):
        """Duplicate instances when a specific number of duplicates is provided."""
        dataset_orig = [[1,1], [0,0], [4.5, 3]]
        num_duplicates = [0,0,3]
        expected_result =  [[1,1], [0,0], [4.5, 3], [4.5, 3], [4.5, 3], [4.5, 3]]
        assert duplicate_dataset(dataset_orig, num_duplicates) == expected_result


class TestToyDataset1AxiomsChallenges:

    def test_paper_example(self):
        """Test example with 16 points as presented in the paper"""
        low_div_data, high_div_data  = create_toy_dataset1_axioms_challenges(16)

        low_div_data_expected = [(0, 0)] * 4 + [(1, 0)] * 4 + [(0, 1)] * 4 + [(1, 1)] * 4

        # use Counter because order doesn't matter
        assert Counter(low_div_data) == Counter(low_div_data_expected)

        high_div_expected = [(0,0), (0, 1/3), (0, 2/3), (0,1),
                            (1/3, 0), (1/3, 1/3), (1/3, 2/3), (1/3,1),
                            (2/3, 0), (2/3, 1/3), (2/3, 2/3), (2/3,1),
                            (1, 0), (1, 1/3), (1, 2/3), (1,1),
                            ]

        assert Counter(high_div_data) == Counter(high_div_expected)


    def test_small_number_points(self):
        low_div_data, high_div_data  = create_toy_dataset1_axioms_challenges(1)
        assert low_div_data == [(0.0,0.0)]
        assert high_div_data == [(0.0,0.0)]
