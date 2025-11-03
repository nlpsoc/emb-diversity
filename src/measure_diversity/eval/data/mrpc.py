"""
Load MRPC paraphrases from CSV file.
"""
import csv
from typing import List
from pathlib import Path


def get_data_path() -> Path:
    """
    Get the path to the mrpc_paraphrases.csv file.

    Returns:
        Path to the CSV file in the same directory as this module
    """
    return Path(__file__).parent.parents[3] / "pilot_datasets/mrpc_paraphrases.csv"


def load_original_sentences() -> List[str]:
    """
    Load original sentences (original_sentence1 column) from the MRPC paraphrases CSV.

    Returns:
        List of original sentences as strings
    """
    csv_path = get_data_path()

    sentences = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentences.append(row['original_sentence1'])

    return sentences


def load_paraphrases() -> List[str]:
    """
    Load paraphrased sentences (paraphrase1 column) from the MRPC paraphrases CSV.

    Returns:
        List of paraphrased sentences as strings
    """
    csv_path = get_data_path()

    paraphrases = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            paraphrases.append(row['paraphrase1'])

    return paraphrases


def load_original_and_paraphrases() -> tuple[List[str], List[str]]:
    """
    Load both original sentences and paraphrases from the MRPC paraphrases CSV.

    Returns:
        Tuple of (original_sentences, paraphrases) as lists of strings
    """
    csv_path = get_data_path()

    originals = []
    paraphrases = []

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            originals.append(row['original_sentence1'])
            paraphrases.append(row['paraphrase1'])

    return originals, paraphrases


if __name__ == "__main__":
    # Example usage
    originals = load_original_sentences()
    paraphrases = load_paraphrases()

    print(f"Loaded {len(originals)} original sentences")
    print(f"Loaded {len(paraphrases)} paraphrases")

    print("\nFirst original sentence:")
    print(originals[0])

    print("\nFirst paraphrase:")
    print(paraphrases[0])
