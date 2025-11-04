"""
    use 1k sample from s1 paper https://arxiv.org/abs/2501.19393
    https://huggingface.co/simplescaling/s1.1-32B
"""
from datasets import load_dataset
import pandas as pd
import os

def get_s1_gemini_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/s1K
    dataset = load_dataset("simplescaling/s1K-1.1")["train"]
    # extract thinking_trajectories
    # thinking_trajectories = [sample["thinking_trajectories"][0] for sample in dataset]
    thinking_trajectories = [sample["gemini_attempt"] for sample in dataset]
    return thinking_trajectories


def get_s1_deepseek_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/s1K-1.1
    dataset = load_dataset("simplescaling/s1K-1.1")["train"]
    # extract deepseek_thinking_trajectory
    # deepseek_thinking_trajectories = [sample["deepseek_thinking_trajectory"] for sample in dataset]
    deepseek_thinking_trajectories = [sample["deepseek_attempt"] for sample in dataset]

    return deepseek_thinking_trajectories

def get_aime_I_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/aime-1k
    dataset = load_dataset("opencompass/AIME2025", 'AIME2025-I')["test"]
    # extract response
    responses = [sample["question"] for sample in dataset]
    return responses

def get_math500_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/math500
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    # extract problem
    problems = [sample["solution"] for sample in dataset]
    return problems

def get_math500_s1_responses(n_entries: int = None) -> list[str]:
    """
    Load model responses from math500_s1_responses.csv

    Args:
        n_entries: Number of entries to load. If None, loads all entries.

    Returns:
        List of model responses
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "math500_s1_responses.csv")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Get the specified number of entries
    if n_entries is not None:
        df = df.head(n_entries)

    # Extract model_response column
    responses = df["model_response"].tolist()
    return responses

def get_math500_s1_1_responses(n_entries: int = None) -> list[str]:
    """
    Load model responses from math500_s1.1_responses.csv

    Args:
        n_entries: Number of entries to load. If None, loads all entries.

    Returns:
        List of model responses
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "math500_s1.1_responses.csv")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Get the specified number of entries
    if n_entries is not None:
        df = df.head(n_entries)

    # Extract model_response column
    responses = df["model_response"].tolist()
    return responses