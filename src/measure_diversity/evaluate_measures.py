from typing import List, Callable, Sequence, Union
import numpy as np

def evaluate_almost_same(
        datasets: List[Sequence[Sequence[float]]],
        measures: List[Callable[[Sequence[float]], float]],
        dataset_names: List[str] | None = None
) -> None:
    """

    :param datasets: List of datasets that should get similar scores in terms of diversity.
                                     Each dataset is a sequence of points (vectors).
                                     Dataset i should have strictly lower diversity than dataset i+1.
    :param measures: List of diversity measure functions to evaluate.
                 Each function should take a dataset and return a diversity score.
    :param dataset_names: Optional list of names for the datasets. If None, uses "Dataset 0", "Dataset 1", etc.
    :return:
    """
    check_input(datasets, measures)
    dataset_names = get_dataset_names(datasets, dataset_names)

    print()
    print("=" * 80)
    print("DIVERSITY MEASURE ALMOST-SAME EVALUATION")
    print("=" * 80)
    print(f"Datasets: {len(datasets)} (expected ~ equal diversity)")
    print(f"Measures: {len(measures)}")
    print()

    measure_names, scores = calculate_diversity_scores(datasets, measures, dataset_names)

    # Evaluate "almost same" for each measure
    print("ALMOST-SAME EVALUATION (within max(ATOL, RTOL*|mean|)):")
    print("-" * 60)
    ATOL = 1e-3  # at least absolute tolerance of 0.001
    RTOL = 0.05  # otherwise 5% of the mean for that measure,
    #              TODO: is this the best way to do this? Chose this way since we don't know the ranges for measures ...

    passing = []
    failing = []

    for name, row in zip(measure_names, scores):
        if any(s is None for s in row):
            print(f"{name.ljust(25)} | FAILED (errors in computation)")
            continue

        vals = [float(s) for s in row if s is not None]
        mean_val = sum(vals) / len(vals)
        tol = max(ATOL, RTOL * abs(mean_val))

        violations = []
        for idx in range(1, len(vals)):

            if abs(vals[idx] - vals[idx-1]) > tol:
                violations.append((dataset_names[idx], vals[idx]))

        if not violations:
            passing.append(name)
            print(f"{name.ljust(25)} | ✓ PASS (mean={mean_val:.6f}, tol={tol:.6g})")
        else:
            failing.append(name)
            viol_str = "; ".join([f"{dn}:{v:.6f}" for dn, v in violations])
            print(f"{name.ljust(25)} | ✗ FAIL (mean={mean_val:.6f}, tol={tol:.6g}; outliers: {viol_str})")

    print()
    print("SUMMARY:")
    print("-" * 60)
    valid = [mn for mn, row in zip(measure_names, scores) if not any(s is None for s in row)]
    print(f"Total measures: {len(measures)}")
    print(f"Valid measures (no errors): {len(valid)}")
    print(f"Almost-same measures: {len(passing)}")
    if valid:
        rate = len(passing) / len(valid) * 100
        print(f"Success rate: {len(passing)}/{len(valid)} = {rate:.1f}%")
    else:
        print("Success rate: N/A (no valid measures)")
    if passing:
        print(f"Passing measures: {', '.join(passing)}")
    if failing:
        print(f"Failing measures: {', '.join(failing)}")
    print("=" * 80)


def evaluate_monotone_order(
        datasets: List[Sequence[Sequence[float]]],
        measures: List[Callable[[Sequence[Sequence[float]]], float]],
        dataset_names: List[str] | None = None
) -> None:
    """
    Evaluate whether diversity measures conform to a strict monotone ordering of datasets.

    Args:
        datasets: List of datasets in strict increasing order of diversity.
                                     Each dataset is a sequence of points (vectors).
                                     Dataset i should have strictly lower diversity than dataset i+1.
        measures: List of diversity measure functions to evaluate.
                 Each function should take a dataset and return a diversity score.
        dataset_names: Optional list of names for the datasets. If None, uses "Dataset 0", "Dataset 1", etc.

    Prints:
        Results showing which measures conform to the expected ordering.
    """
    check_input(datasets, measures)
    dataset_names = get_dataset_names(datasets, dataset_names)

    print()
    print("=" * 80)
    print("DIVERSITY MEASURE MONOTONICITY EVALUATION")
    print("=" * 80)
    print(f"Datasets: {len(datasets)} (expected in strict increasing order of diversity)")
    print(f"Measures: {len(measures)}")
    print()

    # Calculate scores for all measures on all datasets
    measure_names, scores = calculate_diversity_scores(datasets, measures, dataset_names)

    # Evaluate monotonicity for each measure
    print()
    print("MONOTONICITY EVALUATION:")
    print("-" * 60)

    for i, (measure_name, measure_scores) in enumerate(zip(measure_names, scores)):
        # Skip if any scores are None
        if any(score is None for score in measure_scores):
            print(f"{measure_name.ljust(25)} | FAILED (errors in computation)")
            continue

        # Check strict monotonicity: score[i] < score[i+1] for all i
        violations = []
        is_monotonic = True

        for j in range(len(measure_scores) - 1):
            if measure_scores[j] >= measure_scores[j + 1]:
                violations.append((j, j + 1))
                is_monotonic = False

        if is_monotonic:
            status = "✓ PASS"
            print(f"{measure_name.ljust(25)} | {status}")
        else:
            status = "✗ FAIL"
            violation_str = ", ".join([f"{dataset_names[v[0]]}≥{dataset_names[v[1]]}" for v in violations])
            print(f"{measure_name.ljust(25)} | {status} (violations: {violation_str})")

    print()

    # Summary statistics
    print("SUMMARY:")
    print("-" * 60)

    # Count valid measures (no computation errors)
    valid_measures = [i for i, scores_list in enumerate(scores)
                      if not any(score is None for score in scores_list)]

    # Count passing measures (valid and monotonic)
    passing_measures = []
    for i in valid_measures:
        measure_scores = scores[i]
        is_monotonic = all(measure_scores[j] < measure_scores[j + 1]
                           for j in range(len(measure_scores) - 1))
        if is_monotonic:
            passing_measures.append(i)

    print(f"Total measures: {len(measures)}")
    print(f"Valid measures (no errors): {len(valid_measures)}")
    print(f"Monotonic measures: {len(passing_measures)}")
    if len(valid_measures) > 0:
        print(
            f"Success rate: {len(passing_measures)}/{len(valid_measures)} = {len(passing_measures) / len(valid_measures) * 100:.1f}%" if valid_measures else "Success rate: N/A")
    else:
        print("Success rate: N/A (no valid measures)")

    if passing_measures:
        print(f"Passing measures: {', '.join([measure_names[i] for i in passing_measures])}")

    if len(valid_measures) > len(passing_measures):
        failing_measures = [i for i in valid_measures if i not in passing_measures]
        print(f"Failing measures: {', '.join([measure_names[i] for i in failing_measures])}")

    print("=" * 80)


def calculate_diversity_scores(
        datasets: List[Sequence[Sequence[float]]],
        measures: List[Callable[[Sequence[float]], float]],
        dataset_names: List[str] | None = None
) -> tuple[List[str], List[List[float]]]:
    """
     Calculate diversity scores between datasets for each measure in measures.
    :param dataset_names:
    :param datasets:
    :param measures:
    :return:
    """
    scores = []
    measure_names = []
    for i, measure in enumerate(measures):
        try:
            measure_name = getattr(measure, '__name__', f'measure_{i}')
            measure_names.append(measure_name)

            dataset_scores = []
            for j, dataset in enumerate(datasets):
                try:
                    score = measure(dataset)
                    dataset_scores.append(score)
                except Exception as e:
                    print(f"Error: {measure_name} failed on dataset {j}: {e}")
                    dataset_scores.append(None)

            scores.append(dataset_scores)

        except Exception as e:
            print(f"Error: Failed to evaluate measure {i}: {e}")
            scores.append([None] * len(datasets))
            measure_names.append(f'measure_{i}_FAILED')

    # Print scores table
    print("DIVERSITY SCORES:")
    print("-" * 60)
    header = "Measure".ljust(30) + " | " + " | ".join([name[:25].rjust(25) for name in dataset_names])
    print(header)
    print("-" * len(header))
    for i, (measure_name, measure_scores) in enumerate(zip(measure_names, scores)):
        score_strs = []
        for score in measure_scores:
            if score is None:
                score_strs.append("ERROR".rjust(25))
            else:
                score_strs.append(f"{score:.4f}".rjust(25))

        row = measure_name.ljust(30) + " | " + " | ".join(score_strs)
        print(row)

    return measure_names, scores


def get_dataset_names(
        datasets: List[Sequence[Sequence[float]]], dataset_names: List[str] | None = None
) -> List[str]:
    # Set default dataset names if not provided
    if dataset_names is None:
        dataset_names = [f"Dataset {i}" for i in range(len(datasets))]
    elif len(dataset_names) != len(datasets):
        print(f"Warning: Expected {len(datasets)} dataset names, got {len(dataset_names)}. Using defaults.")
        dataset_names = [f"Dataset {i}" for i in range(len(datasets))]
    return dataset_names


def check_input(datasets: List[Sequence[Sequence[float]]], measures: List[Callable[[Sequence[float]], float]]):
    """
        Tests edge cases (less than 2 datasets or less than 1 measure)
    :param datasets:
    :param measures:
    :return:
    """
    if len(datasets) < 2:
        raise ValueError("Error: Need at least 2 datasets to compare diversity measures.")
    if not measures:
        raise ValueError("Error: Need at least 1 measure to evaluate")