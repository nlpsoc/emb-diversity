from measure_diversity.eval.data.s1 import get_s1_gemini_dataset, get_s1_deepseek_dataset
from measure_diversity import evaluate_measures
from measure_diversity.measure import (mean_pairwise_distance, distance_dispersion, cluster_inertia_diversity)
from measure_diversity.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
from measure_diversity.plot.umap import plot_umap_comparable

class TestS1Diversity:
    def test_diversity_s1(self):
        s1_gem = get_s1_gemini_dataset()# [:10]
        s1_ds = get_s1_deepseek_dataset()# [:10]

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

        s1g_sem = encode_semantic_sentences(s1_gem)
        s1d_sem = encode_semantic_sentences(s1_ds)

        # Plot UMAP with comparable embeddings (same reduction)
        semantic_plots = plot_umap_comparable(
            {"Gemini": s1g_sem, "DeepSeek": s1d_sem},
            title="S1 Semantic Embeddings"
        )
        for name, plot in semantic_plots.items():
            plot.save(f"s1_semantic_{name.lower()}_umap.png", dpi=300)

        s1g_style = encode_style_sentences(s1_gem)
        s1d_style = encode_style_sentences(s1_ds)


        # Plot UMAP with comparable embeddings (same reduction)
        semantic_plots = plot_umap_comparable(
            {"Gemini": s1g_style, "DeepSeek": s1d_style},
            title="S1 Semantic Embeddings"
        )
        for name, plot in semantic_plots.items():
            plot.save(f"s1_style_{name.lower()}_umap.png", dpi=300)

        evaluate_measures.evaluate_monotone_order(
            [s1g_style, s1d_style], measures,
            dataset_names=["s1 gemini", "s1 deepseek"])

        evaluate_measures.evaluate_monotone_order(
            [s1g_sem, s1d_sem], measures,
            dataset_names=["s1 gemini", "s1 deepseek"])