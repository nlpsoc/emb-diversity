from measure_diversity.eval.data.s1 import get_s1_gemini_dataset, get_s1_deepseek_dataset, get_aime_I_dataset, get_math500_dataset, get_model_responses, get_math500_s1_1_responses, get_math500_s1_reasoning, get_math500_s1_reasoning, get_math500_s1_1_reasoning
from measure_diversity import evaluate_measures
from measure_diversity.measure import (mean_pairwise_distance, distance_dispersion, cluster_inertia_diversity)
from measure_diversity.embeddings.SBERT import encode_style_sentences, encode_semantic_sentences
from measure_diversity.plot.umap import plot_umap_comparable

# import pydevd_pycharm
# pydevd_pycharm.settrace('hpcs05', port=5678, stdout_to_server=True,
#                         stderr_to_server=True)


class TestS1Diversity:
    def test_diversity_s1(self):
        s1_gem = get_s1_gemini_dataset()[:500]
        s1_ds = get_s1_deepseek_dataset()[:500]

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

        print("encoding semantics")
        # aime_sem = encode_semantic_sentences(aimeI)
        s1g_sem = encode_semantic_sentences(s1_gem)
        s1d_sem = encode_semantic_sentences(s1_ds)
        # math500_sem = encode_semantic_sentences(math500)

        # Plot UMAP with comparable embeddings (same reduction)
        # Using default colors (colors=None) - can be customized with colors={"Gemini": "#FF0000", ...}
        semantic_plots = plot_umap_comparable(
            {"Gemini": s1g_sem, "DeepSeek": s1d_sem, # "AIME-I": aime_sem, "Math500": math500_sem
             },
            title="Semantic Representations",
            # colors={"Gemini": "#1f77b4", "DeepSeek": "#ff7f0e"},
            # colors={"Gemini": "#14345A", "DeepSeek": "#E6A823"},
            # colors={"Gemini": "#1F4E79", "DeepSeek": "#D28E00"},
            colors={"Gemini": "#003B44", "DeepSeek": "#E07A7A"},
            # colors={"Gemini": "#7EC9D3", "DeepSeek": "#5A1414"},
            save_dir=".",
            dpi=300
        )


        print("encoding style")
        s1g_style = encode_style_sentences(s1_gem)
        s1d_style = encode_style_sentences(s1_ds)


        # Plot UMAP with comparable embeddings (same reduction)
        # Default plotnine colors shown explicitly - modify as needed
        style_plots = plot_umap_comparable(
            {"Gemini": s1g_style, "DeepSeek": s1d_style , # "AIME-I": aime_style, "Math500": math500_style
             },
            title="Style Representations",
            colors={"Gemini": "#003B44", "DeepSeek": "#E07A7A"},
            save_dir=".",
            dpi=300
        )

        evaluate_measures.evaluate_monotone_order(
            [s1g_style, s1d_style], measures,
            dataset_names=["s1 gemini", "s1 deepseek"])

        evaluate_measures.evaluate_monotone_order(
            [s1g_sem, s1d_sem], measures,
            dataset_names=["s1 gemini", "s1 deepseek"])

    def test_generation_diversity(self):
        # Load first 270 entries from CSV files
        math500_s1 = get_math500_s1_reasoning(n_entries=270)
        math500_s1_1 = get_math500_s1_1_reasoning(n_entries=270)

        # Load first 270 entries from HuggingFace datasets
        s1_gem = get_s1_gemini_dataset()[:270]
        s1_ds = get_s1_deepseek_dataset()[:270]

        print("Encoding semantics for all datasets")
        math500_s1_sem = encode_semantic_sentences(math500_s1)
        math500_s1_1_sem = encode_semantic_sentences(math500_s1_1)
        s1g_sem = encode_semantic_sentences(s1_gem)
        s1d_sem = encode_semantic_sentences(s1_ds)

        # Create combined UMAP plot for semantic representations
        semantic_plots = plot_umap_comparable(
            {
                "Qwen-DeepSeek": math500_s1_1_sem,
                "Qwen-Gemini": math500_s1_sem,
                "DeepSeek": s1d_sem,
                "Gemini": s1g_sem,
            },
            colors={"Qwen-DeepSeek": "#000000", "Qwen-Gemini": "#0072B2", "DeepSeek": "#D55E00", "Gemini": "#CC79A7"},
            title="Semantic Representations",
            save_dir=".",
            dpi=300
        )

        print("Encoding style for all datasets")
        math500_s1_style = encode_style_sentences(math500_s1)
        math500_s1_1_style = encode_style_sentences(math500_s1_1)
        s1g_style = encode_style_sentences(s1_gem)
        s1d_style = encode_style_sentences(s1_ds)

        # Create combined UMAP plot for style representations
        style_plots = plot_umap_comparable(
            {
                "Qwen-DeepSeek": math500_s1_1_style,
                "Qwen-Gemini": math500_s1_style,
                "DeepSeek": s1d_style,
                "Gemini": s1g_style,
            },
            title="Style Representations",
            colors={"Qwen-DeepSeek": "#000000", "Qwen-Gemini": "#0072B2", "DeepSeek": "#D55E00", "Gemini": "#CC79A7"},
            save_dir=".",
            dpi=300
        )

if __name__ == "__main__":
    test = TestS1Diversity()
    print("Testing S1 Diversity Measures")
    # test.test_diversity_s1()
    test.test_generation_diversity()
