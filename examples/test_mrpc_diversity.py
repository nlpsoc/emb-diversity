from measure_diversity.eval.data.mrpc import load_original_sentences, load_paraphrases
from measure_diversity import evaluate_measures
from measure_diversity.measure import (mean_pairwise_distance, distance_dispersion, cluster_inertia_diversity)
from measure_diversity.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
from measure_diversity.plot.umap import plot_umap_comparable


class TestMRPCDiversity:
    def test_diversity_mrpc(self):
        # Load MRPC data
        originals = load_original_sentences()
        paraphrases = load_paraphrases()

        measures = [
            lambda data: mean_pairwise_distance(data, metric="euclidean"),
            lambda data: mean_pairwise_distance(data, metric="cosine"),
            lambda data: distance_dispersion(data, metric="cosine"),
            cluster_inertia_diversity
        ]
        # Add names to lambda functions for better display
        measures[0].__name__ = "mean_pairwise_euclidean"
        measures[1].__name__ = "mean_pairwise_cosine"
        measures[2].__name__ = "distance_dispersion_cosine"

        print("Encoding semantics for MRPC data")
        originals_sem = encode_semantic_sentences(originals)
        paraphrases_sem = encode_semantic_sentences(paraphrases)

        # Plot UMAP with comparable embeddings (same reduction)
        semantic_plots = plot_umap_comparable(
            {"Original": originals_sem, "Mistral Rephrase": paraphrases_sem},
            title="Semantic Representations",
            save_dir=".",
            dpi=300
        )

        print("Encoding style for MRPC data")
        originals_style = encode_style_sentences(originals)
        paraphrases_style = encode_style_sentences(paraphrases)

        # Plot UMAP with comparable embeddings (same reduction)
        style_plots = plot_umap_comparable(
            {"Original": originals_style, "Mistral Rephrase": paraphrases_style},
            title="Style Representations",
            save_dir=".",
            dpi=300
        )

        # Evaluate diversity measures
        print("\n=== Style Embeddings Diversity ===")
        evaluate_measures.evaluate_monotone_order(
            [originals_style, paraphrases_style],
            measures,
            dataset_names=["MRPC Original", "MRPC Paraphrase"]
        )

        print("\n=== Semantic Embeddings Diversity ===")
        evaluate_measures.evaluate_monotone_order(
            [originals_sem, paraphrases_sem],
            measures,
            dataset_names=["MRPC Original", "MRPC Paraphrase"]
        )


if __name__ == "__main__":
    test = TestMRPCDiversity()
    print("Testing MRPC Diversity Measures")
    test.test_diversity_mrpc()
