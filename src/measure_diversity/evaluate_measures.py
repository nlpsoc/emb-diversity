from typing import List, Callable, Sequence, Union
import numpy as np

def evaluate_monotone_order(
        strict_monotone_dataset_order: List[Sequence[Sequence[float]]],
        measures: List[Callable[[Sequence[Sequence[float]]], float]],
        dataset_names: List[str] | None = None
) -> None:
    """
    Evaluate whether diversity measures conform to a strict monotone ordering of datasets.

    Args:
        strict_monotone_dataset_order: List of datasets in strict increasing order of diversity.
                                     Each dataset is a sequence of points (vectors).
                                     Dataset i should have strictly lower diversity than dataset i+1.
        measures: List of diversity measure functions to evaluate.
                 Each function should take a dataset and return a diversity score.
        dataset_names: Optional list of names for the datasets. If None, uses "Dataset 0", "Dataset 1", etc.

    Prints:
        Results showing which measures conform to the expected ordering.
    """
    if len(strict_monotone_dataset_order) < 2:
        print("Error: Need at least 2 datasets to evaluate monotone order")
        return

    if not measures:
        print("Error: Need at least 1 measure to evaluate")
        return

    n_datasets = len(strict_monotone_dataset_order)
    n_measures = len(measures)

    # Set default dataset names if not provided
    if dataset_names is None:
        dataset_names = [f"Dataset {i}" for i in range(n_datasets)]
    elif len(dataset_names) != n_datasets:
        print(f"Warning: Expected {n_datasets} dataset names, got {len(dataset_names)}. Using defaults.")
        dataset_names = [f"Dataset {i}" for i in range(n_datasets)]

    print()
    print("=" * 80)
    print("DIVERSITY MEASURE MONOTONICITY EVALUATION")
    print("=" * 80)
    print(f"Datasets: {n_datasets} (expected in strict increasing order of diversity)")
    print(f"Measures: {n_measures}")
    print()

    # Calculate scores for all measures on all datasets
    scores = []
    measure_names = []

    for i, measure in enumerate(measures):
        try:
            measure_name = getattr(measure, '__name__', f'measure_{i}')
            measure_names.append(measure_name)

            dataset_scores = []
            for j, dataset in enumerate(strict_monotone_dataset_order):
                try:
                    score = measure(dataset)
                    dataset_scores.append(score)
                except Exception as e:
                    print(f"Error: {measure_name} failed on dataset {j}: {e}")
                    dataset_scores.append(None)

            scores.append(dataset_scores)

        except Exception as e:
            print(f"Error: Failed to evaluate measure {i}: {e}")
            scores.append([None] * n_datasets)
            measure_names.append(f'measure_{i}_FAILED')

    # Print scores table
    print("DIVERSITY SCORES:")
    print("-" * 60)
    header = "Measure".ljust(25) + " | " + " | ".join([name[:15].rjust(15) for name in dataset_names])
    print(header)
    print("-" * len(header))

    for i, (measure_name, measure_scores) in enumerate(zip(measure_names, scores)):
        score_strs = []
        for score in measure_scores:
            if score is None:
                score_strs.append("ERROR".rjust(15))
            else:
                score_strs.append(f"{score:.4f}".rjust(15))

        row = measure_name.ljust(25) + " | " + " | ".join(score_strs)
        print(row)

    print()

    # Evaluate monotonicity for each measure
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

    print(f"Total measures: {n_measures}")
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