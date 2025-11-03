"""
Generate responses for MATH-500 dataset using simplescaling s1 and s1.1 models.
"""
import csv
import os
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_math500_dataset() -> List[Dict[str, str]]:
    """
    Load the MATH-500 dataset.

    Returns:
        List of dictionaries containing math problems
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Extract problems from the dataset
    data = []
    for item in dataset:
        data.append({
            'problem': item['problem'],
            'solution': item.get('solution', ''),
            'answer': item.get('answer', ''),
            'subject': item.get('subject', ''),
            'level': item.get('level', '')
        })

    return data


def generate_response(
    problem: str,
    model,
    tokenizer,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate a response for a given math problem.

    Args:
        problem: The input math problem
        model: The loaded language model
        tokenizer: The model's tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        The generated response
    """
    # Construct the prompt
    prompt = f"{problem}"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True  # Enable KV cache for faster generation
        )

    # Decode the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (remove the prompt)
    if problem in full_output:
        response = full_output.split(problem, 1)[-1].strip()
    else:
        response = full_output.strip()

    return response


def load_existing_results(output_csv: str) -> tuple[List[Dict], int]:
    """
    Load existing results from CSV if it exists.

    Args:
        output_csv: Path to the CSV file

    Returns:
        Tuple of (list of existing results, number of completed items)
    """
    if not os.path.exists(output_csv):
        return [], 0

    results = []
    with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append(row)

    print(f"Found {len(results)} existing results in {output_csv}")
    return results, len(results)


def save_results(results: List[Dict], output_csv: str) -> None:
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_csv: Path to the output CSV file
    """
    fieldnames = [
        'problem',
        'model_response',
        'ground_truth_solution',
        'ground_truth_answer',
        'subject',
        'level'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def generate_responses_for_math500(
    model_name: str,
    output_csv: str,
    max_samples: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_flash_attention: bool = False,
    save_interval: int = 10,
    resume: bool = True
) -> None:
    """
    Generate responses for all problems in the MATH-500 dataset and save to CSV.

    Args:
        model_name: HuggingFace model identifier
        output_csv: Path to the output CSV file
        max_samples: Maximum number of samples to process (None for all)
        device: Device to run the model on ('cuda' or 'cpu')
        use_flash_attention: Whether to use Flash Attention 2 for faster inference (default: True)
        save_interval: Save results every N samples (default: 10)
        resume: Whether to resume from existing results if found (default: True)
    """
    print(f"Loading MATH-500 dataset...")
    data = load_math500_dataset()

    if max_samples is not None:
        data = data[:max_samples]

    # Check for existing results and resume if requested
    results = []
    start_idx = 0
    if resume:
        results, start_idx = load_existing_results(output_csv)
        if start_idx > 0:
            print(f"Resuming from index {start_idx}...")
        if start_idx >= len(data):
            print(f"All {len(data)} samples already processed!")
            return

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimizations
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    # Add Flash Attention 2 if available and requested
    if use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2 for faster inference")
        except Exception:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile for faster inference...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Could not compile model: {e}")

    print(f"Generating responses for {len(data) - start_idx} problems (starting from {start_idx})...")

    for idx in range(start_idx, len(data)):
        item = data[idx]
        print(f"Processing {idx + 1}/{len(data)}...")

        try:
            # Generate response for the problem
            response = generate_response(
                item['problem'],
                model,
                tokenizer
            )

            results.append({
                'problem': item['problem'],
                'model_response': response,
                'ground_truth_solution': item['solution'],
                'ground_truth_answer': item['answer'],
                'subject': item['subject'],
                'level': item['level']
            })

            # Save periodically
            if (idx + 1) % save_interval == 0 or (idx + 1) == len(data):
                print(f"Saving checkpoint at {idx + 1}/{len(data)}...")
                save_results(results, output_csv)

        except Exception as e:
            print(f"Error processing item {idx + 1}: {e}")
            print(f"Saving progress before exiting...")
            save_results(results, output_csv)
            raise

    print(f"Done! All {len(results)} results saved to {output_csv}")


if __name__ == "__main__":
    # Generate responses for s1-32B model
    print("=" * 80)
    print("Generating responses with s1-32B model...")
    print("=" * 80)
    generate_responses_for_math500(
        model_name="simplescaling/s1-32B",
        output_csv="math500_s1_responses.csv",
        max_samples=None  # Process all samples
    )

    # Generate responses for s1.1-32B model
    print("\n" + "=" * 80)
    print("Generating responses with s1.1-32B model...")
    print("=" * 80)
    generate_responses_for_math500(
        model_name="simplescaling/s1.1-32B",
        output_csv="math500_s1.1_responses.csv",
        max_samples=None  # Process all samples
    )
