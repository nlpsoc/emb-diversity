"""
Generate paraphrases for MRPC dataset using Mistral-7B-Instruct model.
"""
import csv
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_mrpc_dataset(split: str = "train") -> List[Dict[str, str]]:
    """
    Load the MRPC (Microsoft Research Paraphrase Corpus) dataset.

    Args:
        split: Dataset split to load ('train', 'validation', or 'test')

    Returns:
        List of dictionaries containing sentence pairs
    """
    dataset = load_dataset("glue", "mrpc", split=split)

    # Extract sentences from the dataset
    data = []
    for item in dataset:
        data.append({
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'label': item['label']
        })

    return data


def generate_paraphrase(
    sentence: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate a paraphrase for a given sentence using Mistral-7B-Instruct.

    Args:
        sentence: The input sentence to paraphrase
        model: The loaded language model
        tokenizer: The model's tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        The generated paraphrase
    """
    # Construct the prompt with the specified format
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the questions. "
        "USER: For the following paragraph give me a diverse paraphrase of the same "
        "in high quality English language as in sentences on Wikipedia:\n\n"
        f"{sentence}\n\n"
        "ASSISTANT:"
    )

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the paraphrase
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response (after "ASSISTANT:")
    if "ASSISTANT:" in full_output:
        paraphrase = full_output.split("ASSISTANT:")[-1].strip()
    else:
        paraphrase = full_output.strip()

    return paraphrase


def generate_paraphrases_for_mrpc(
    output_csv: str = "mrpc_paraphrases.csv",
    split: str = "train",
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    max_samples: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Generate paraphrases for all sentences in the MRPC dataset and save to CSV.

    Args:
        output_csv: Path to the output CSV file
        split: Dataset split to use ('train', 'validation', or 'test')
        model_name: HuggingFace model identifier
        max_samples: Maximum number of samples to process (None for all)
        device: Device to run the model on ('cuda' or 'cpu')
    """
    print(f"Loading MRPC dataset ({split} split)...")
    data = load_mrpc_dataset(split)

    if max_samples is not None:
        data = data[:max_samples]

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()

    print(f"Generating paraphrases for {len(data)} sentence pairs...")

    results = []
    for idx, item in enumerate(data):
        print(f"Processing {idx + 1}/{len(data)}...")

        # Generate paraphrases for both sentences
        paraphrase1 = generate_paraphrase(
            item['sentence1'],
            model,
            tokenizer
        )

        paraphrase2 = generate_paraphrase(
            item['sentence2'],
            model,
            tokenizer
        )

        results.append({
            'original_sentence1': item['sentence1'],
            'paraphrase1': paraphrase1,
            'original_sentence2': item['sentence2'],
            'paraphrase2': paraphrase2,
            'label': item['label']
        })

    # Save to CSV
    print(f"Saving results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'original_sentence1',
            'paraphrase1',
            'original_sentence2',
            'paraphrase2',
            'label'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Results saved to {output_csv}")


if __name__ == "__main__":
    # Example usage: generate paraphrases for first 10 samples of train split
    generate_paraphrases_for_mrpc(
        output_csv="mrpc_paraphrases.csv",
        split="train",
        max_samples=1000  # Remove or set to None to process all samples
    )
