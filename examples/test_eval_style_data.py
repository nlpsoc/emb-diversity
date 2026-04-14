from embediver import evaluate_measures
from embediver import (mean_pw_dist, dist_dispersion, cluster_inertia,
                                       convex_hull_volume)
import numpy as np

class TestOnSTEL:

    def test_eval_on_stel(self):
        """
            Test the evaluate_measures function on style datasets
        """
        # Load more and less diverse formal/informal dataset
        from embediver.eval.data.STEL import create_formal_diverse, create_formal_only
        from embediver.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
        only_formal_text = create_formal_only()
        half_formal_text = create_formal_diverse(formal_share=0.5)

        measures = [
            lambda data: mean_pw_dist(data, metric="euclidean"),
            lambda data: mean_pw_dist(data, metric="cosine"),
            lambda data: dist_dispersion(data, metric="cosine"),
            # convex_hull_volume, --> this takes too long
            cluster_inertia
        ]

        # Add names to lambda functions for better display
        measures[0].__name__ = "mean_pairwise_euclidean"
        measures[1].__name__ = "mean_pairwise_cosine"
        measures[2].__name__ = "dist_dispersion_cosine"

        only_formal_style_vectors = encode_style_sentences(only_formal_text)
        half_formal_style_vectors = encode_style_sentences(half_formal_text)

        only_formal_semantic_vectors = encode_semantic_sentences(only_formal_text)
        half_formal_semantic_vectors = encode_semantic_sentences(half_formal_text)

        evaluate_measures.evaluate_monotone_order(
            [only_formal_style_vectors, half_formal_style_vectors], measures,
            dataset_names=["only formal", "half formal half informal"])

        evaluate_measures.evaluate_almost_same(
            [only_formal_semantic_vectors, half_formal_semantic_vectors], measures,
            dataset_names=["only formal semantic", "half formal half informal semantic"],)

    def test_variety(self):
        from embediver.eval.data.synthstel import get_low_variety, get_high_variety
        from embediver.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
        low_variety = get_low_variety()
        high_variety = get_high_variety()

        measures = [
            lambda data: mean_pw_dist(data, metric="euclidean"),
            lambda data: mean_pw_dist(data, metric="cosine"),
            lambda data: dist_dispersion(data, metric="cosine"),
            cluster_inertia
        ]

        # Add names to lambda functions for better display
        measures[0].__name__ = "mean_pairwise_euclidean"
        measures[1].__name__ = "mean_pairwise_cosine"
        measures[2].__name__ = "dist_dispersion_cosine"

        low_variety_sv = encode_style_sentences(low_variety)
        high_variety_sv = encode_style_sentences(high_variety)

        evaluate_measures.evaluate_monotone_order(
            [low_variety_sv, high_variety_sv], measures,
            dataset_names=["low variety style", "high variety style"])


    def test_balance(self):
        from embediver.eval.data.synthstel import get_low_balance, get_high_balance
        from embediver.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
        low_balance = get_low_balance()
        high_balance = get_high_balance()

        measures = [
            lambda data: mean_pw_dist(data, metric="euclidean"),
            lambda data: mean_pw_dist(data, metric="cosine"),
            lambda data: dist_dispersion(data, metric="cosine"),
            cluster_inertia
        ]

        # Add names to lambda functions for better display
        measures[0].__name__ = "mean_pairwise_euclidean"
        measures[1].__name__ = "mean_pairwise_cosine"
        measures[2].__name__ = "dist_dispersion_cosine"

        low_balance_sv = encode_style_sentences(low_balance)
        high_balance_sv = encode_style_sentences(high_balance)

        evaluate_measures.evaluate_monotone_order(
            [low_balance_sv, high_balance_sv], measures,
            dataset_names=["low balance style", "high balance style"])

    def test_disparity(self):
        from embediver.eval.data.synthstel import get_low_disparity, get_high_disparity
        from embediver.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
        low_disparity = get_low_disparity()
        high_disparity = get_high_disparity()

        measures = [
            lambda data: mean_pw_dist(data, metric="euclidean"),
            lambda data: mean_pw_dist(data, metric="cosine"),
            lambda data: dist_dispersion(data, metric="cosine"),
            cluster_inertia
        ]

        # Add names to lambda functions for better display
        measures[0].__name__ = "mean_pairwise_euclidean"
        measures[1].__name__ = "mean_pairwise_cosine"
        measures[2].__name__ = "dist_dispersion_cosine"

        low_disparity_sv = encode_style_sentences(low_disparity)
        high_disparity_sv = encode_style_sentences(high_disparity)

        evaluate_measures.evaluate_monotone_order(
            [low_disparity_sv, high_disparity_sv], measures,
            dataset_names=["low disparity style", "high disparity style"])