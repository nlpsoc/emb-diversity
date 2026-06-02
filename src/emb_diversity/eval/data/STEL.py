"""
    access STEL formality pairs from pilot datasets
"""
import re
import pandas as pd
import numpy as np
from emb_diversity.utility import project_root


STEL_path = (project_root.find() / "pilot_datasets" / "style_diversity" / "original_data" /
             "_quad_stel-dimensions_formal-815_complex-815.tsv")

def get_formal_informal_STEL_pairs(tsv_file_path=STEL_path):
    """
    Extract unique formal-informal sentence pairs from STEL file

    Args:
        tsv_file_path (str): Path to the TSV file

    Returns:
        pd.DataFrame: DataFrame with 'formal' and 'informal' columns containing unique pairs
    """
    # Read the TSV file
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Dictionary to store sentences by their unique ID
    sentences_by_id = {}

    # Process each row
    for _, row in df.iterrows():
        # Extract all unique identifiers from the ID field
        # Pattern matches i-XXXX or f-XXXX in order
        matches = re.findall(r'([if])-(\d+)',  row['ID'])

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
    """
        for each sentence pair from STEL formaility dimension,
            randomly select one of the two sentences with probability formal_share
    :param formal_share:
    :return:
    """
    formal_pairs = get_formal_informal_STEL_pairs()

    # Generate random choices for each pair
    choose_formal = np.random.random(len(formal_pairs)) < formal_share

    # Select sentences based on random choices
    sentences = np.where(choose_formal, formal_pairs['formal'], formal_pairs['informal'])

    return sentences.tolist()

def create_formal_only():
    formal_pairs = get_formal_informal_STEL_pairs()
    return formal_pairs['formal'].tolist()

