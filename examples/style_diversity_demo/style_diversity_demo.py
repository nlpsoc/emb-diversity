"""Demonstrate the diversity measures on style datasets.

A runnable example: it builds style datasets that *should* score low vs. high
on variety, balance, disparity, or formality diversity, and uses the package's
``measure_diversity`` convenience function with its default measures to check
that each measure scores the "high" dataset above the "low" one.

Install the package together with the ``datasets`` HuggingFace loader, then run
this script from its own folder::

    pip install emb-diversity datasets
    python style_diversity_demo.py

The STEL loader reads the ``_quad_stel-dimensions_formal-815_complex-815.tsv``
file shipped alongside this script; the SynthSTEL loader pulls a HuggingFace
dataset via the ``datasets`` package.
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset

from emb_diversity import measure_diversity


# ── STEL formality pairs (local file shipped with this script) ────────

STEL_path = Path(__file__).resolve().parent / "_quad_stel-dimensions_formal-815_complex-815.tsv"


def get_formal_informal_STEL_pairs(tsv_file_path=STEL_path):
    """Extract unique formal-informal sentence pairs from the STEL file.

    Args:
        tsv_file_path (str): Path to the TSV file

    Returns:
        pd.DataFrame: DataFrame with 'formal' and 'informal' columns containing unique pairs
    """
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Dictionary to store sentences by their unique ID
    sentences_by_id = {}

    for _, row in df.iterrows():
        # Extract all unique identifiers from the ID field
        # Pattern matches i-XXXX or f-XXXX in order
        matches = re.findall(r'([if])-(\d+)', row['ID'])

        # Get the sentence columns (excluding metadata columns)
        sentence_columns = ['Anchor 1', 'Anchor 2', 'Alternative 1.1', 'Alternative 1.2']

        # Map each sentence to its formality type based on position and ID pattern
        for i, (formality_type, unique_id) in enumerate(matches):
            # Get the corresponding sentence (assumes order matches)
            if unique_id not in sentences_by_id:
                sentences_by_id[unique_id] = {}

            sentences_by_id[unique_id][formality_type] = row[sentence_columns[i]]

    # Create pairs from sentences that have both formal and informal versions
    pairs = [
        {'formal': sentences['f'], 'informal': sentences['i'], 'id': uid}
        for uid, sentences in sentences_by_id.items()
        if 'f' in sentences and 'i' in sentences
    ]

    return pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)


def create_formal_diverse(formal_share=0.5):
    """For each STEL formality pair, randomly pick the formal sentence with
    probability *formal_share*, else the informal one."""
    formal_pairs = get_formal_informal_STEL_pairs()
    choose_formal = np.random.random(len(formal_pairs)) < formal_share
    sentences = np.where(choose_formal, formal_pairs['formal'], formal_pairs['informal'])
    return sentences.tolist()


def create_formal_only():
    formal_pairs = get_formal_informal_STEL_pairs()
    return formal_pairs['formal'].tolist()


# ── SynthSTEL style features (HuggingFace dataset) ────────────────────

NBR_SENT_PER_FEATURE = 90  # Synthstel generated 90-10 sentences (train-test split) for each style feature using GPT-4

orthographic = {
    'All Lower Case / Proper Capitalization': 'positive',
    'All Upper Case / Proper Capitalization': 'positive',
    'With uppercase letters / Without uppercase letters': 'negative',
    'Sentence With a Few Misspelled Words / Normal Sentence': 'positive',
    'With Emojis / No Emojis': 'positive',
    'Text Emojis / No Emojis': 'positive',
}

phonetic = {
    'With number substitution / Without number substitution': 'positive',
}

syntactic = {
    'Active / Passive': 'negative',
}

lexical = {
    'With function words / Less frequent function words': 'negative',
    'With pronouns / Less frequent pronouns': 'negative',
    'With common verbs / Less frequent common verbs': 'negative',
    'With determiners / Less frequent determiners': 'negative',
    'Long average word length / Short average word length': 'positive',
    'With articles / Less frequent articles': "negative",
}

morphological = {
    'With nominalizations / Without nominalizations': 'positive',
    'With contractions / Without contractions': 'positive',
}

discourse = {
    'Complex / Simple': 'positive',
    'Formal / Informal': 'negative',
}

all_features = discourse | orthographic | lexical | morphological | syntactic | phonetic


@lru_cache(maxsize=1)
def get_synthstel_pair_dicts():
    return load_dataset("StyleDistance/synthstel")["train"]


def get_feature_sentences(feature: str) -> List[str]:
    dataset = get_synthstel_pair_dicts()
    filtered = dataset.filter(lambda row: row["feature"] == feature)
    return filtered[all_features[feature]]


def get_low_variety() -> List[str]:
    """For 6 features get all example sentences."""
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    return [sentence for feature in features for sentence in get_feature_sentences(feature)]


def get_high_variety():
    """For all features, get 6*90 example sentences."""
    features = list(all_features.keys())
    total_sentences = 6 * NBR_SENT_PER_FEATURE
    per_feature_sentences = int(total_sentences / len(features))
    remainder = total_sentences % per_feature_sentences

    result = []
    for idx, feature in enumerate(features):
        count = per_feature_sentences + (1 if idx < remainder else 0)
        feature_sentences = get_feature_sentences(feature)
        result.extend(feature_sentences[:count])
    return result


def get_low_balance():
    """Get 90% from one feature."""
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    sentences = (get_feature_sentences(features[0])[:85] +
                 [get_feature_sentences(feature)[0] for feature in features[1:]])
    return sentences


def get_high_balance():
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    per_feature_count = NBR_SENT_PER_FEATURE // len(features)  # number of sentences per feature
    remainder = NBR_SENT_PER_FEATURE % len(features)
    sentences = []
    for idx, feature in enumerate(features):
        count = per_feature_count + (1 if idx < remainder else 0)
        feature_sentences = get_feature_sentences(feature)[:count]
        sentences.extend(feature_sentences)
    return sentences


def get_low_disparity():
    """Get all orthographic features, all sentences."""
    return [sentence for ortho in orthographic for sentence in get_feature_sentences(ortho)]


def get_high_disparity():
    """Get one feature from each category."""
    assert len(orthographic) == 6
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    return [sentence for feature in features for sentence in get_feature_sentences(feature)]


# ── Demonstrations ────────────────────────────────────────────────────

def compare(label, low_texts, high_texts, diversity_axis="style"):
    """Run the default diversity measures on a low- and a high-diversity dataset
    and print, per measure, whether the high dataset scores above the low one.

    Uses :func:`emb_diversity.measure_diversity` with its default measure
    selection — it embeds the texts on *diversity_axis* itself (with caching).
    """
    print("=" * 80)
    print(f"{label}  (diversity_axis={diversity_axis!r})")
    print("=" * 80)

    low = measure_diversity(low_texts, diversity_axis=diversity_axis)
    high = measure_diversity(high_texts, diversity_axis=diversity_axis)

    for name in low:
        low_value = low[name]["value"]
        high_value = high[name]["value"]
        ok = "✓" if high_value > low_value else "✗"
        print(f"  {ok} {name:<14}  low={low_value:.4f}  high={high_value:.4f}  "
              f"(Δ={high_value - low_value:+.4f})")
    print()


def demo_formality():
    """Style diversity rises when formal and informal sentences are mixed.

    The same formality contrast should be (almost) invisible to a *semantic*
    embedding, so the comparison is also run on the semantic axis as a control.
    """
    only_formal = create_formal_only()
    half_formal = create_formal_diverse(formal_share=0.5)

    compare("Formality: only formal vs. half formal/half informal",
            only_formal, half_formal, diversity_axis="style")
    compare("Formality (semantic control: expect little change)",
            only_formal, half_formal, diversity_axis="semantic")


def demo_variety():
    """Style diversity rises as more distinct style features are present."""
    compare("Variety: few style features vs. many",
            get_low_variety(), get_high_variety())


def demo_balance():
    """Style diversity rises as style features are more evenly balanced."""
    compare("Balance: one dominant feature vs. evenly balanced",
            get_low_balance(), get_high_balance())


def demo_disparity():
    """Style diversity rises as the style features span more distinct categories."""
    compare("Disparity: one category vs. many distinct categories",
            get_low_disparity(), get_high_disparity())


def main():
    demo_formality()
    demo_variety()
    demo_balance()
    demo_disparity()


if __name__ == "__main__":
    main()
