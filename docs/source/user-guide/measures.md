# Available Measures

emb-diversity provides 22 diversity measures.
All measures accept either raw text (list of strings) or pre-computed embeddings (array of vectors).
The three measures marked **default** below (`graph_entropy`, `vendi_score`, `mean_pw_dist`) are the ones that run when you don't name any.

## Interpreting the scores

Each diversity measure returns a `{"value": float, "parameters": {...}, "version":
str}` dict; the `value` is the diversity score, `parameters` records the
configuration used to produce it, and `version` is the installed
`emb-diversity` package version that computed it. We list a few things to keep
in mind when reading those scores:

- **Higher = more diverse.** For every measure, a larger value means a
  more diverse set and a smaller value a less diverse one. This holds even for
  measures that return negative values: `log_determinant` and `energy` are
  always negative, so a *less negative* value (closer to zero) means more diverse. Keep in mind that this reflects diversity as the measure defines it, not an objective property of the dataset itself.

- **Scores are not normalized.** There is no fixed scale: some measures
  are bounded (e.g. `vendi_score` lies in `[1, n]`; cosine-based distances in
  `[0, 2]`), others are unbounded. A score's absolute magnitude is only
  meaningful relative to other scores from the *same* measure.

- **Scores are sensitive to dataset size.** Some measures change with the number
  of items `n`,  so you can't directly compare datasets of different sizes. To compare datasets, use the same measure and subsample them to the same `n`.
 
- **Measures are sensitive to the embedding model.** Different embedding models will yield different scores, so you should always use the same embedding model when comparing datasets.

## Measures

| Function | Brief Description |
|---|---|
| {func}`mean_pw_dist <emb_diversity.measures.mean_pw_dist.mean_pw_dist>` | **(default)** - Average pairwise distance between all data points |
| {func}`sum_pairwise_dist <emb_diversity.measures.sum_pairwise_dist.sum_pairwise_dist>` | Sum of all pairwise distances |
| {func}`chamfer_dist <emb_diversity.measures.chamfer_dist.chamfer_dist>` | Average nearest-neighbour distance across all data points |
| {func}`knn <emb_diversity.measures.knn.knn>` | Average k-th-nearest-neighbour distance across all data points |
| {func}`energy <emb_diversity.measures.energy.energy>` | Negative mean pairwise repulsive energy between all data points (equally-charged-particle model) |
| {func}`diameter <emb_diversity.measures.diameter.diameter>` | Largest distance between any two data points |
| {func}`bottleneck <emb_diversity.measures.bottleneck.bottleneck>` | Smallest distance between any two data points |
| {func}`sum_diameter <emb_diversity.measures.sum_diameter.sum_diameter>` | For each point, find its farthest neighbour; sum those distances |
| {func}`sum_bottleneck <emb_diversity.measures.sum_bottleneck.sum_bottleneck>` | For each point, find its nearest neighbour; sum those distances |
| {func}`convex_hull_volume_3d <emb_diversity.measures.convex_hull_volume_3d.convex_hull_volume_3d>` | Volume of the convex hull after UMAP-projecting embeddings to 3D |
| {func}`geo_mean_std <emb_diversity.measures.geo_mean_std.geo_mean_std>` | Geometric mean of per-dimension standard deviations |
| {func}`span_centroid <emb_diversity.measures.span_centroid.span_centroid>` | Span relative to the centroid |
| {func}`span_medoid <emb_diversity.measures.span_medoid.span_medoid>` | Span relative to the medoid |
| {func}`cluster_inertia <emb_diversity.measures.cluster_inertia.cluster_inertia>` | Sum of squared distances to cluster centres |
| {func}`log_determinant <emb_diversity.measures.log_determinant.log_determinant>` | Log-determinant of the kernel matrix (LDD) |
| {func}`vendi_score <emb_diversity.measures.vendi_score.vendi_score>` | **(default)** - Effective number of distinct items |
| {func}`renyi_entropy <emb_diversity.measures.renyi_entropy.renyi_entropy>` | Renyi Kernel Entropy (RKE) / Matrix-based Renyi entropy |
| {func}`dcscore <emb_diversity.measures.dcscore.dcscore>` | Score based on self-similarity with softmax normalisation |
| {func}`bins_entropy <emb_diversity.measures.bins_entropy.bins_entropy>` | Shannon entropy over a 2D UMAP/PCA-projected histogram |
| {func}`mst_dispersion <emb_diversity.measures.mst_dispersion.mst_dispersion>` | Total edge weight of the minimum spanning tree |
| {func}`graph_entropy <emb_diversity.measures.graph_entropy.graph_entropy>` | **(default)** - Sum of per-point entropy of normalized pairwise-distance distributions over a complete graph |
| {func}`hamdiv <emb_diversity.measures.hamdiv.hamdiv>` | Approximate length of the shortest Hamiltonian circuit through all data points |

## Measure descriptions

### Mean Pairwise Distance — {func}`mean_pw_dist <emb_diversity.measures.mean_pw_dist.mean_pw_dist>`

**Intuition.** Averages the distance between every pair of samples. Higher means
samples sit further apart on average, i.e. more diverse. Under cosine distance,
this is sometimes called (1 − Self-CosSim) in its similarity form
— the two are linearly related.

**Computation.** Compute all pairwise distances between distinct samples and
return their mean.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Guy Tevet and Jonathan Berant. 2021. Evaluating the Evaluation of Diversity in Natural Language Generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 326–346, Online. Association for Computational Linguistics.
- Tianhui Zhang, Bei Peng, and Danushka Bollegala. 2024. Improving Diversity of Commonsense Generation by Large Language Models via In-Context Learning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9226–9242, Miami, Florida, USA. Association for Computational Linguistics.
- Miranda, Brando, Alycia Lee, Sudharsan Sundar, Allison Casasola, Rylan Schaeffer, Elyas Obbad, and Sanmi Koyejo. "Beyond scale: The diversity coefficient as a data quality metric for variability in natural language data." arXiv preprint arXiv:2306.13840 (2023).
- Cox, Samuel Rhys, et al. "Directed diversity: Leveraging language embedding distances for collective creativity in crowd ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021.

### Sum Pairwise Distance — {func}`sum_pairwise_dist <emb_diversity.measures.sum_pairwise_dist.sum_pairwise_dist>`

**Intuition.** Sums all pairwise distances to capture the overall spread of the
dataset. Also known as Max Dispersion or DistSum. Grows with `n`, so it is only comparable across datasets of the same size.

**Computation.** Compute all pairwise distances and return their sum.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.
- Yang, Yuming, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. "Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18530-18549. 2025.

### Chamfer Distance — {func}`chamfer_dist <emb_diversity.measures.chamfer_dist.chamfer_dist>`

**Intuition.** Adapted from point-cloud comparison in computer vision to a
single dataset: averages each sample's distance to its nearest neighbour. This
reflects local compactness: tight interior clusters yield small values, more
dispersed datasets yield larger ones.

**Computation.** For each sample, find its nearest-neighbour distance (excluding
itself), then return the mean across all samples.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.
- Zhang, Tianhui, Bei Peng, and Danushka Bollegala. "Evaluating the Evaluation of Diversity in Commonsense Generation." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), edited by Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar. Association for Computational Linguistics, 2025. https://aclanthology.org/2025.acl-long.1181/.

### KNN Distance — {func}`knn <emb_diversity.measures.knn.knn>`

**Intuition.** Generalizes Chamfer Distance: instead of each sample's nearest
neighbour, look at its *k*-th nearest neighbour (`k=1` is exactly Chamfer
Distance). Larger k-th-nearest-neighbour distances indicate a more evenly spread
dataset.

**Computation.** For each sample, find the distance to its k-th nearest neighbour
(excluding itself), then return the mean across all samples.

**Parameters.** `k` (default `2`; must be ≥ 1; requires at least `k + 1` samples);
`metric` (default `"cosine"`).

**References.**
- Yang, Yuming, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. "Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18530-18549. 2025.

### Energy — {func}`energy <emb_diversity.measures.energy.energy>`

**Intuition.** Borrowed from classical mechanics: treats samples as equally
charged particles that repel each other in inverse proportion to their distance
raised to the power `gamma`. More spread-out configurations have higher (less
negative) potential energy, i.e. higher diversity. The value is always ≤ 0; a
value closer to 0 means more diverse.

**Computation.** Compute all pairwise distances (floored at a small `epsilon` so
duplicates don't blow up the reciprocal), raise each to the power `gamma` and
take the reciprocal, then return the negative mean of those pairwise energies.

**Parameters.** `gamma` (default `1.0` — Coulomb-like repulsion); `epsilon`
(default `1e-12`); `metric` (default `"cosine"`).

**References.**
- Velikonivtsev, Fedor, Mikhail Mironov, and Liudmila Prokhorenkova. "Challenges of generating structurally diverse graphs." Advances in Neural Information Processing Systems 37 (2024): 57993-58022.
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.

### Diameter — {func}`diameter <emb_diversity.measures.diameter.diameter>`

**Intuition.** The largest distance between any two samples — the "width" of the
point cloud.

**Computation.** Compute all pairwise distances and return the maximum.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
- Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. "How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules." arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

### Bottleneck — {func}`bottleneck <emb_diversity.measures.bottleneck.bottleneck>`

**Intuition.** The smallest distance between any two samples — the most
redundant pair in the dataset.

**Computation.** Compute all pairwise distances and return the minimum.

**Caveat.** As a stand-alone measure this can be misleading: a single pair of
near-duplicates collapses the value regardless of the rest of the distribution.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
- Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. "How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules." arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

### Sum Diameter — {func}`sum_diameter <emb_diversity.measures.sum_diameter.sum_diameter>`

**Intuition.** For each sample, take its distance to its farthest neighbour, then
sum these. Aggregating over every point makes it less sensitive to a single
extreme pair than Diameter.

**Computation.** For each sample, find its maximum distance to any other sample,
then sum these maxima (or average them if `normalize_by_n=True`).

**Parameters.** `metric` (default `"cosine"`); `normalize_by_n` (default `False`).

**References.**
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
- Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. "How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules." arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

### Sum Bottleneck — {func}`sum_bottleneck <emb_diversity.measures.sum_bottleneck.sum_bottleneck>`

**Intuition.** For each sample, take its distance to its nearest neighbour, then
sum these. Aggregating over every point makes it less sensitive to any single
(near-)duplicate pair than Bottleneck, but it still mainly reflects *local*
density rather than overall spread.

**Computation.** For each sample, find its minimum distance to any other sample,
then sum these minima (or average them if `normalize_by_n=True`).

**Parameters.** `metric` (default `"cosine"`); `normalize_by_n` (default `False`).

**References.**
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
- Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. "How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules." arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

### Convex Hull Volume — {func}`convex_hull_volume_3d <emb_diversity.measures.convex_hull_volume_3d.convex_hull_volume_3d>`

**Intuition.** Treats each embedding as a point and computes the volume of the
smallest convex region (convex hull) containing all of them. A larger hull
volume means the samples span a larger region, i.e. more diverse.

**Computation.** Project embeddings to 3D with UMAP (used as-is if the input is
already 3D; direct computation in the full embedding space is intractable, as
the number of hull facets grows exponentially with dimension), compute the
convex hull of the projected points (Quickhull), and return its volume (`0.0` if
the projected points are coplanar).

**Parameters.** `random_state` (default `42`, seeds UMAP).

**Caveat.** Because the volume is computed in a nonlinear, data-dependent UMAP
projection, values are only comparable within a single experiment using the same
`random_state` — not across different datasets or separate UMAP fits.

**References.**
- Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.

### Geometric Mean of Standard Deviations — {func}`geo_mean_std <emb_diversity.measures.geo_mean_std.geo_mean_std>`

**Intuition.** Treats each embedding dimension as an independent axis of
variation and summarizes the dataset's spread as the geometric mean of the
per-dimension standard deviations — the "radius" of the axis-aligned ellipsoid
spanned by those per-axis standard deviations.

**Computation.** Compute the standard deviation along each embedding dimension
and return their geometric mean.

**Caveat.** As a geometric mean, a single near-constant dimension can drag the
whole value toward 0 even if every other dimension is well spread. The value
also scales with the magnitude of the input vectors, so it is not comparable
across differently-scaled embeddings.

**References.**
- Lai, Yi-An, et al. "Diversity, density, and homogeneity: Quantitative characteristic metrics for text collections." Proceedings of the Twelfth Language Resources and Evaluation Conference. 2020.

### Span (Centroid) — {func}`span_centroid <emb_diversity.measures.span_centroid.span_centroid>`

**Intuition.** The centroid represents the "centre" of the dataset. Span
(Centroid) quantifies how far, on average, samples deviate from it, using a high
percentile rather than the mean for robustness to outliers.

**Computation.** Compute the centroid (mean embedding), then each sample's
distance to it, then return the given percentile of those distances.

**Parameters.** `percentile` (default `90.0`); `metric` (default `"cosine"`).

**References.**
- Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

### Span (Medoid) — {func}`span_medoid <emb_diversity.measures.span_medoid.span_medoid>`

**Intuition.** Like Span (Centroid), but uses the medoid — the sample with the
smallest total distance to all others — as the reference point instead of the
mean. Less sensitive to outliers than the centroid.

**Computation.** Find the medoid (the sample minimizing the sum of distances to
all others), then return the mean distance from all samples to it.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

### Cluster Inertia — {func}`cluster_inertia <emb_diversity.measures.cluster_inertia.cluster_inertia>`

**Intuition.** Borrowed from the physical "moment of inertia": partitions
samples into k-means clusters and measures how far, on average, samples sit from
their cluster centres. Higher inertia means samples resist collapsing into a few
tight clusters, indicating higher diversity.

**Computation.** Run k-means with `n_clusters` clusters and return the total
within-cluster sum of squared distances to each point's assigned cluster centre.

**Parameters.** `n_clusters` (default `200`; automatically reduced to `n - 1` if
fewer data points than clusters are given).

**References.**
- Yang, Yuming, Yang Nan, Junjie Ye, et al. "Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric." arXiv:2502.17184. Preprint, arXiv, February 28, 2025. https://doi.org/10.48550/arXiv.2502.17184.
- Du, Wenchao, and Alan W. Black. "Boosting Dialog Response Generation." In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, edited by Anna Korhonen, David Traum, and Lluís Màrquez. Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/P19-1005.

### Log-Determinant Diversity (LDD) — {func}`log_determinant <emb_diversity.measures.log_determinant.log_determinant>`

**Intuition.** The determinant of a kernel matrix equals the squared volume
spanned by the underlying feature vectors. Redundant samples occupy a smaller
subspace and yield a lower determinant, so a higher (less negative)
log-determinant indicates greater diversity.

**Computation.** Build a kernel matrix, add a small jitter `eps` to the diagonal
for numerical stability (the cosine kernel is singular whenever there are more
samples than embedding dimensions), and return its log-determinant via Cholesky
factorisation.

**Parameters.** `kernel_type` (default `"cs"`, a cosine-similarity-like kernel on
L2-normalized vectors; also `"rbf"`, `"lap"`, `"poly"`); `tau`; `eps` (default
`1e-6`); `use_cholesky` (default `True`).

**References.**
- Kulesza, A., & Taskar, B. (2012). Determinantal Point Processes for Machine Learning. Found. Trends Mach. Learn., 5, 123-286.
- Wang, Peiqi, Yikang Shen, Zhen Guo, Matthew Stallone, Yoon Kim, Polina Golland, and Rameswar Panda. "Diversity measurement and subset selection for instruction tuning datasets." arXiv preprint arXiv:2402.02318 (2024).

### Vendi Score — {func}`vendi_score <emb_diversity.measures.vendi_score.vendi_score>`

**Intuition.** The "effective number of unique elements" in a dataset, derived
from the entropy of the eigenvalue spectrum of a similarity matrix. Reaches its
maximum (`n`) when all samples are orthogonal and its minimum (`1`) when all
samples are identical. The `q` parameter generalizes the score to the
Hill-number family (sometimes called Vendi-*q*); `q=1` (the default) recovers
the original Shannon-entropy-based Vendi Score.

**Computation.** Build a similarity matrix (dot product on optionally
L2-normalized vectors, i.e. cosine similarity by default), normalize it to unit
trace, and return the exponential of the order-`q` Rényi entropy of its
eigenvalues.

**Parameters.** `q` (default `1.0`); `normalize` (default `True`); `use_dual`
(default `True`, an equivalent formulation that avoids building the full
similarity matrix and is efficient when the embedding dimension is smaller than
`n`).

**References.**
- Friedman, Dan, and Adji Bousso Dieng. "The Vendi Score: A Diversity Evaluation Metric for Machine Learning." Transactions on Machine Learning Research (2023).
- Pasarkar, A. P., & Dieng, A. B. (2023). Cousins of the vendi score: A family of similarity-based diversity metrics for science and machine learning. arXiv preprint arXiv:2310.12952.

### Rényi Kernel Entropy (RKE) — {func}`renyi_entropy <emb_diversity.measures.renyi_entropy.renyi_entropy>`

**Intuition.** Estimates the effective number of modes or clusters in the data by
treating the normalized eigenvalue spectrum of a kernel/similarity matrix as a
probability distribution and computing its Rényi entropy. More uniformly
distributed samples yield higher entropy, i.e. more diversity.

**Computation.** Build a PSD kernel matrix, normalize it to unit trace so its
eigenvalues form a probability distribution, and return the order-`alpha` Rényi
entropy of those eigenvalues. At `alpha=2` (the default) this has a closed form
that avoids eigendecomposition.

**Parameters.** `alpha` (default `2.0`); `kernel_type` (default `"cs"`, a
cosine-similarity-like kernel on optionally L2-normalized vectors; also
`"rbf"`, `"lap"`, `"poly"`); `tau`; `normalize` (default `True`).

**References.**
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
- Jalali, Mohammad, Cheuk Ting Li, and Farzan Farnia. "An information-theoretic evaluation of generative models in learning multi-modal distributions." Advances in Neural Information Processing Systems 36 (2023): 9931-9943.

### DCScore — {func}`dcscore <emb_diversity.measures.dcscore.dcscore>`

**Intuition.** Frames diversity as an n-class self-classification problem: each
sample is its own class, and diversity is how easily each sample can be told
apart from the rest under a softmax-based classifier. More distinct samples are
more easily "classified" into their own class.

**Computation.** Build a kernel/similarity matrix, apply a row-wise softmax
(temperature `tau`) to get a class-probability matrix, and return its trace —
the total probability mass each sample assigns to itself.

**Parameters.** `kernel_type` (default `"cs"`, a cosine-similarity-like kernel on
optionally L2-normalized vectors; also `"rbf"`, `"lap"`, `"poly"`); `tau`
(default `1.0`); `normalize` (default `True`).

**References.**
- Zhu, Yuchang, Huizhe Zhang, Bingzhe Wu, Jintang Li, Zibin Zheng, Peilin Zhao, Liang Chen, and Yatao Bian. "Measuring diversity in synthetic datasets." arXiv preprint arXiv:2502.08512 (2025).

### Bins Entropy — {func}`bins_entropy <emb_diversity.measures.bins_entropy.bins_entropy>`

**Intuition.** Quantifies how evenly samples are spread across a discretized 2D
projection of the embedding space. Datasets whose samples occupy many bins with
similar frequencies have higher entropy, i.e. more diversity.

**Computation.** Project embeddings to 2D (UMAP by default, or PCA), bin the
projected points into an `n_bins_x` × `n_bins_y` grid, and compute the Shannon
entropy of the occupied-bin distribution. By default the entropy is normalized
by `log(min(n, total_bins))` (the `"effective"` mode), which keeps the result in
`[0, 1]` regardless of how many bins are actually occupied; a `"bins"`
normalization mode (dividing by `log(total_bins)` instead) is also available but
is not the default.

**Parameters.** `n_bins_x` / `n_bins_y` (default `5` × `5`); `projection`
(default `"umap"`, or `"pca"`); `normalize` (default `True`); `normalization`
(default `"effective"`, or `"bins"`).

**References.**
- Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.
- Yang, Yuming, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. "Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18530-18549. 2025.

### MST Dispersion — {func}`mst_dispersion <emb_diversity.measures.mst_dispersion.mst_dispersion>`

**Intuition.** Builds a complete weighted graph over samples (edge weight =
pairwise distance) and computes its Minimum Spanning Tree (MST). The total MST
edge weight is the minimum total cost needed to connect all samples, capturing
global dispersion while ignoring redundant internal connections.

**Computation.** Build the complete distance graph, compute its MST, and return
the sum of the MST's edge weights. This total is not normalized by the number of
samples, so it grows with `n`.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Cox, Samuel Rhys, et al. "Directed diversity: Leveraging language embedding distances for collective creativity in crowd ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021.
- Atwal, Tevin, Chan Nam Tieu, Yefeng Yuan, Zhan Shi, Yuhong Liu, and Liang Cheng. "Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs." arXiv preprint arXiv:2507.18055 (2025).

### Graph Entropy — {func}`graph_entropy <emb_diversity.measures.graph_entropy.graph_entropy>`

**Intuition.** Builds a complete weighted graph over samples (edge weight =
pairwise distance) and measures how evenly each node's distances to all others
are distributed. A more uniform local distance distribution (higher per-node
entropy) indicates higher diversity.

**Computation.** For each node, normalize its distances to all other nodes into
a local probability distribution and compute its local Shannon entropy; return
the sum of all nodes' local entropies. Requires at least 3 samples — with only
2, every node has a single neighbour and the local entropy is degenerately 0.

**Parameters.** `metric` (default `"cosine"`).

**References.**
- Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.

### Hamiltonian Diversity (HamDiv) — {func}`hamdiv <emb_diversity.measures.hamdiv.hamdiv>`

**Intuition.** Treats dataset diversity as the length of the shortest tour that
visits every sample exactly once (a Hamiltonian circuit / Travelling Salesman
tour) over the complete distance graph. Widely separated embeddings yield longer
tours, i.e. higher diversity.

**Computation.** Build the complete weighted distance graph, approximate the
shortest Hamiltonian circuit with the chosen TSP heuristic, and return the
tour's total length.

**Parameters.** `solver` (default `"christofides"`, or `"greedy"`); `metric`
(default `"cosine"`).

**Caveat.** Exact computation is NP-hard (equivalent to solving the TSP), so this
implementation always returns an approximate solution via a heuristic solver —
practical for small-to-medium datasets rather than very large ones.

**References.**
- Hu, Xiuyuan, et al. "Hamiltonian diversity: effectively measuring molecular diversity by shortest hamiltonian circuits." Journal of Cheminformatics 16.1 (2024): 94.
- Mironov, Mikhail, and Liudmila Prokhorenkova. "Measuring Diversity: Axioms and Challenges." arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.

## Taxonomy

The measures above can also be grouped into four families, following the
taxonomy proposed in:

> Cantao Su, Anna Wegmann, Esther Ploeger, and Dong Nguyen. 2026. *Measuring
> data diversity with embeddings: A taxonomy and benchmark for diversity
> measures.* Under review.

This grouping isn't required knowledge for using `emb-diversity` — it's included
as background. When [adding a new measure](adding-a-measure.md) to the package,
it's worth considering which family it fits (and whether that family already has
similar coverage), but this is a helpful design consideration rather than a
requirement.

| Family | Measures |
|---|---|
| **Distance-based**<br>Uses raw pairwise distances/similarities directly, without building additional structures (graphs, cluster assignments, geometric constructs). | {func}`mean_pw_dist <emb_diversity.measures.mean_pw_dist.mean_pw_dist>`, {func}`sum_pairwise_dist <emb_diversity.measures.sum_pairwise_dist.sum_pairwise_dist>`, {func}`chamfer_dist <emb_diversity.measures.chamfer_dist.chamfer_dist>`, {func}`knn <emb_diversity.measures.knn.knn>`, {func}`energy <emb_diversity.measures.energy.energy>` |
| **Geometry-based**<br>Treats embeddings as a point cloud in dimension-`d` space and quantifies diversity through geometric properties — extent, volume, or spread around a representative point. | {func}`diameter <emb_diversity.measures.diameter.diameter>`, {func}`bottleneck <emb_diversity.measures.bottleneck.bottleneck>`, {func}`sum_diameter <emb_diversity.measures.sum_diameter.sum_diameter>`, {func}`sum_bottleneck <emb_diversity.measures.sum_bottleneck.sum_bottleneck>`, {func}`convex_hull_volume_3d <emb_diversity.measures.convex_hull_volume_3d.convex_hull_volume_3d>`, {func}`geo_mean_std <emb_diversity.measures.geo_mean_std.geo_mean_std>`, {func}`span_centroid <emb_diversity.measures.span_centroid.span_centroid>`, {func}`span_medoid <emb_diversity.measures.span_medoid.span_medoid>`, {func}`cluster_inertia <emb_diversity.measures.cluster_inertia.cluster_inertia>`, {func}`log_determinant <emb_diversity.measures.log_determinant.log_determinant>` |
| **Distribution-based**<br>Converts the dataset into a probability distribution (e.g. an eigenvalue spectrum, or a binned histogram) and quantifies diversity through entropy-based functionals of that distribution. | {func}`vendi_score <emb_diversity.measures.vendi_score.vendi_score>`, {func}`renyi_entropy <emb_diversity.measures.renyi_entropy.renyi_entropy>`, {func}`dcscore <emb_diversity.measures.dcscore.dcscore>`, {func}`bins_entropy <emb_diversity.measures.bins_entropy.bins_entropy>` |
| **Graph-theory-based**<br>Represents the dataset as a complete weighted graph (nodes = samples, edges = pairwise distances) and quantifies diversity from the structure of that graph. | {func}`mst_dispersion <emb_diversity.measures.mst_dispersion.mst_dispersion>`, {func}`graph_entropy <emb_diversity.measures.graph_entropy.graph_entropy>`, {func}`hamdiv <emb_diversity.measures.hamdiv.hamdiv>` |

Each measure module in `src/emb_diversity/measures/` marks its family with a
`### <Family>-Based Diversity Measure` comment placed right after the file's
imports (e.g. `### Distance-Based Diversity Measure`). When contributing a new
built-in measure, add this comment for the family it belongs to — see
[Adding New Measures](https://github.com/nlpsoc/Diversity-Measurement#adding-new-measures)
in the README for the full steps.
