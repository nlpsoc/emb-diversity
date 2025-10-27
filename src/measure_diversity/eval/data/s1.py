"""
    use 1k sample from s1 paper https://arxiv.org/abs/2501.19393
"""
from datasets import load_dataset

def get_s1_gemini_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/s1K
    dataset =  load_dataset("simplescaling/s1K")["train"]
    # extract thinking_trajectories
    thinking_trajectories = [sample["thinking_trajectories"][0] for sample in dataset]
    return thinking_trajectories


def get_s1_deepseek_dataset() -> list[str]:
    # load and return https://huggingface.co/datasets/simplescaling/s1K-1.1
    dataset = load_dataset("simplescaling/s1K-1.1")["train"]
    # extract deepseek_thinking_trajectory
    deepseek_thinking_trajectories = [sample["deepseek_thinking_trajectory"] for sample in dataset]
    return deepseek_thinking_trajectories