"""
    get low/high variety, balance and disparity datasets for style
"""

from typing import Dict, List
from datasets import load_dataset
import datasets
from functools import lru_cache

""" ----------------- category and feature assignment -------------- """

orthographic = {
	'All Lower Case / Proper Capitalization': 'positive',
	'All Upper Case / Proper Capitalization': 'positive',
	'With uppercase letters / Without uppercase letters': 'negative',
	# 'With digits / Less frequent digits': '',
	'Sentence With a Few Misspelled Words / Normal Sentence': 'positive',
	'With Emojis / No Emojis': 'positive',
	'Text Emojis / No Emojis': 'positive',
	# 'With frequent punctuation / Less Frequent punctuation',
}

phonetic = {
	'With number substitution / Without number substitution': 'positive',
}

syntactic = {
	'Active / Passive': 'negative',
	# 'With conjunctions / Less frequent conjunctions': ,
	# 'With prepositions / Less frequent prepositions'
}

lexical = {
	'With function words / Less frequent function words': 'negative',
	'With pronouns / Less frequent pronouns': 'negative',
	'With common verbs / Less frequent common verbs': 'negative',
	'With determiners / Less frequent determiners': 'negative',
	'Long average word length / Short average word length': 'positive',
	'With articles / Less frequent articles': "negative"
}

morphological = {
	'With nominalizations / Without nominalizations': 'positive',
	'With contractions / Without contractions': 'positive',
}

discourse = {
	'Complex / Simple': 'positive',
	'Formal / Informal': 'negative'
}

all_features = discourse | orthographic | lexical | morphological | syntactic | phonetic

""" ----------------- functions ------------------------------------ """

@lru_cache(maxsize=1)
def get_synthstel_pair_dicts() -> datasets.arrow_dataset.Dataset:
    return load_dataset("StyleDistance/synthstel")["train"]

def get_feature_sentences(feature: str) -> List[str]:
    dataset = get_synthstel_pair_dicts()
    filtered = dataset.filter(lambda row: row["feature"] == feature)
    return filtered[all_features[feature]]


def get_low_variety() -> List[str]:
    """
        for 6 features get all example sentences
    :return: 
    """
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]

    return [sentence for feature in features for sentence in get_feature_sentences(feature)]


def get_high_variety():
    """
    for all features, get 6*90 example sentences
    :return:
    """
    features = list(all_features.keys())
    total_sentences = 6 * 90
    per_feature_sentences = int(total_sentences / len(features))
    remainder = total_sentences % per_feature_sentences

    result = []
    for idx, feature in enumerate(features):
        count = per_feature_sentences + (1 if idx < remainder else 0)
        feature_sentences = get_feature_sentences(feature)
        result.extend(feature_sentences[:count])

    return result

def get_low_balance():
    """
        get 90% from one feature
    :return:
    """
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    sentences = (get_feature_sentences(features[0])[:85] +
                      [get_feature_sentences(feature)[0] for feature in features[1:]])
    return sentences

def get_high_balance():
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]
    per_feature_count = 90 // len(features)
    remainder = 90 % len(features)
    sentences = []
    for idx, feature in enumerate(features):
        count = per_feature_count + (1 if idx < remainder else 0)
        feature_sentences = get_feature_sentences(feature)[:count]
        sentences.extend(feature_sentences)
    return sentences

def get_low_disparity():
    """
        get all orthographic features all sentences
    :return:
    """
    return [sentence for ortho in orthographic for sentence in get_feature_sentences(ortho)]

def get_high_disparity():
    """
        get features
    :return:
    """
    assert len(orthographic) == 6
    features = [next(iter(orthographic)), next(iter(phonetic)), next(iter(syntactic)), next(iter(lexical)),
                next(iter(morphological)), next(iter(discourse))]

    return [sentence for feature in features for sentence in get_feature_sentences(feature)]
