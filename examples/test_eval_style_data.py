"""Style-data demonstrations for the diversity measures.

These are example/demonstration checks, not part of the pytest suite — the
``examples/`` directory is excluded from collection (see ``norecursedirs`` in
``pyproject.toml``). Each test builds style datasets that *should* score low
vs. high on variety, balance, or disparity, embeds them with the package's
embedding API, and sanity-checks the measures.

The data loaders live here rather than in the installable package: the STEL
loader reads local ``pilot_datasets/`` files and the SynthSTEL loader pulls a
HuggingFace dataset (the ``datasets`` package, a dev-only dependency). Keeping
them out of ``emb_diversity`` lets the published package stay free of both.
"""
import re
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset

from emb_diversity import (mean_pw_dist, dist_dispersion, cluster_inertia,
                           convex_hull_volume_2d, evaluate_measures)
from emb_diversity.embed import embed_texts
from emb_diversity.utility import project_root


# ── STEL formality pairs (local pilot dataset) ────────────────────────

STEL_path = (project_root.find() / "pilot_datasets" / "style_diversity" / "original_data" /
             "_quad_stel-dimensions_formal-815_complex-815.tsv")


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

def _style_measures():
    """Build the measure list used by every demonstration, with display names."""
    measures = [
        lambda data: mean_pw_dist(data, metric="euclidean"),
        lambda data: mean_pw_dist(data, metric="cosine"),
        lambda data: dist_dispersion(data, metric="cosine"),
        # convex_hull_volume_2d, --> this takes too long (UMAP fit on high-dim embeddings)
        cluster_inertia,
    ]
    measures[0].__name__ = "mean_pairwise_euclidean"
    measures[1].__name__ = "mean_pairwise_cosine"
    measures[2].__name__ = "dist_dispersion_cosine"
    return measures


class TestOnSTEL:

    def test_eval_on_stel(self):
        """Test the evaluate_measures function on style datasets."""
        # Load more and less diverse formal/informal dataset
        only_formal_text = create_formal_only()
        half_formal_text = create_formal_diverse(formal_share=0.5)

        measures = _style_measures()

        only_formal_style_vectors = embed_texts(only_formal_text, diversity_axis="style")
        half_formal_style_vectors = embed_texts(half_formal_text, diversity_axis="style")

        only_formal_semantic_vectors = embed_texts(only_formal_text, diversity_axis="semantic")
        half_formal_semantic_vectors = embed_texts(half_formal_text, diversity_axis="semantic")

        evaluate_measures.evaluate_monotone_order(
            [only_formal_style_vectors, half_formal_style_vectors], measures,
            dataset_names=["only formal", "half formal half informal"])

        evaluate_measures.evaluate_almost_same(
            [only_formal_semantic_vectors, half_formal_semantic_vectors], measures,
            dataset_names=["only formal semantic", "half formal half informal semantic"],)

    def test_variety(self):
        low_variety = get_low_variety()
        high_variety = get_high_variety()

        measures = _style_measures()

        low_variety_sv = embed_texts(low_variety, diversity_axis="style")
        high_variety_sv = embed_texts(high_variety, diversity_axis="style")

        evaluate_measures.evaluate_monotone_order(
            [low_variety_sv, high_variety_sv], measures,
            dataset_names=["low variety style", "high variety style"])

    def test_balance(self):
        low_balance = get_low_balance()
        high_balance = get_high_balance()

        measures = _style_measures()

        low_balance_sv = embed_texts(low_balance, diversity_axis="style")
        high_balance_sv = embed_texts(high_balance, diversity_axis="style")

        evaluate_measures.evaluate_monotone_order(
            [low_balance_sv, high_balance_sv], measures,
            dataset_names=["low balance style", "high balance style"])

    def test_disparity(self):
        low_disparity = get_low_disparity()
        high_disparity = get_high_disparity()

        measures = _style_measures()

        low_disparity_sv = embed_texts(low_disparity, diversity_axis="style")
        high_disparity_sv = embed_texts(high_disparity, diversity_axis="style")

        evaluate_measures.evaluate_monotone_order(
            [low_disparity_sv, high_disparity_sv], measures,
            dataset_names=["low disparity style", "high disparity style"])
