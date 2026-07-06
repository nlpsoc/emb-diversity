# Available Measures

emb-diversity provides 22 diversity measures.
All measures accept either raw text (list of strings) or pre-computed embeddings (array of vectors).
The three measures marked **default** below (`graph_entropy`, `vendi_score`, `mean_pw_dist`) are the ones that run when you don't name any.

## Interpreting the scores

Each diversity measure returns a `{"value": float, "parameters": {...}}` dict;
the `value` is the diversity score and `parameters` records the configuration
used to produce it. We list a few things to keep in mind when reading those
scores:

- **Higher = more diverse.** For every measure, a larger value means a
  more diverse set and a smaller value a less diverse one. This holds even for
  measures that return negative values: `log_determinant` and `energy` are
  always negative, so a *less negative* value (closer to zero) means more diverse. Keep in mind that this reflects diversity as the measure defines it, not an objective property of the dataset itself.

- **Scores are not normalized.** There is no fixed scale: some measures
  are bounded (e.g. `vendi_score` lies in `[1, n]`; cosine-based distances in
  `[0, 2]`), others are unbounded. A score's absolute magnitude is only
  meaningful relative to other scores from the *same* measure.

- **Scores are sensitive to dataset size.** Measures change with the number
  of items `n`,  so you can't directly compare datasets of different sizes. To compare datasets, use the same measure and subsample them to the same `n`.
 
- **Measures are sensitive to the embedding model.** Different embedding models will yield different scores, so you should always use the same embedding model when comparing datasets.

## Measures

| Function | Description |
|---|---|
| {func}`mean_pw_dist <emb_diversity.measures.mean_pw_dist.mean_pw_dist>` | Average pairwise distance between all datapoints -- **default** |
| {func}`sum_pairwise_dist <emb_diversity.measures.sum_pairwise_dist.sum_pairwise_dist>` | Sum of all pairwise distances |
| {func}`diameter <emb_diversity.measures.diameter.diameter>` | Largest distance between any two instances |
| {func}`bottleneck <emb_diversity.measures.bottleneck.bottleneck>` | Smallest distance between any two instances |
| {func}`sum_bottleneck <emb_diversity.measures.sum_bottleneck.sum_bottleneck>` | For each point, find its nearest neighbour; sum those distances |
| {func}`sum_diameter <emb_diversity.measures.sum_diameter.sum_diameter>` | For each point, find its farthest neighbour; sum those distances |
| {func}`chamfer_dist <emb_diversity.measures.chamfer_dist.chamfer_dist>` | Average nearest-neighbour distance across all datapoints |
| {func}`knn <emb_diversity.measures.knn.knn>` | Average k-th-nearest-neighbour distance across all datapoints |
| {func}`span_centroid <emb_diversity.measures.span_centroid.span_centroid>` | Distance-based span relative to the centroid (Cox et al., 2021) |
| {func}`span_medoid <emb_diversity.measures.span_medoid.span_medoid>` | Distance-based span relative to the medoid (Cox et al., 2021) |
| {func}`vendi_score <emb_diversity.measures.vendi_score.vendi_score>` | Vendi Score -- effective number of distinct items (Friedman & Dieng, 2023) -- **default** |
| {func}`renyi_entropy <emb_diversity.measures.renyi_entropy.renyi_entropy>` | Renyi Kernel Entropy (RKE) / Matrix-based Renyi entropy |
| {func}`bins_entropy <emb_diversity.measures.bins_entropy.bins_entropy>` | Shannon entropy over a 2D UMAP/PCA-projected histogram |
| {func}`graph_entropy <emb_diversity.measures.graph_entropy.graph_entropy>` | Entropy of the degree distribution of a nearest-neighbour graph -- **default** |
| {func}`convex_hull_volume_2d <emb_diversity.measures.convex_hull_volume_2d.convex_hull_volume_2d>` | Area of the convex hull after UMAP-projecting embeddings to 2D |
| {func}`hamdiv <emb_diversity.measures.hamdiv.hamdiv>` | Length of the shortest Hamiltonian circuit through all points |
| {func}`mst_dispersion <emb_diversity.measures.mst_dispersion.mst_dispersion>` | Total edge weight of the minimum spanning tree |
| {func}`radius <emb_diversity.measures.radius.radius>` | Geometric mean of per-dimension standard deviations |
| {func}`energy <emb_diversity.measures.energy.energy>` | Energy distance between the dataset and a reference distribution |
| {func}`cluster_inertia <emb_diversity.measures.cluster_inertia.cluster_inertia>` | Inertia -- sum of squared distances to cluster centres |
| {func}`dcscore <emb_diversity.measures.dcscore.dcscore>` | DC Score based on self-similarity with softmax normalisation |
| {func}`log_determinant <emb_diversity.measures.log_determinant.log_determinant>` | Log-determinant of the kernel matrix (LDD) |
