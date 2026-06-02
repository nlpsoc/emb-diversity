# Available Measures

emb-diversity provides 20 diversity measures across four categories.
All measures accept either raw text (list of strings) or pre-computed embeddings (array of vectors).

## Distance-based

| Function | Description |
|---|---|
| {func}`mean_pw_dist <emb_diversity.measures.mean_pw_dist.mean_pw_dist>` | Average pairwise distance between all datapoints |
| {func}`dist_dispersion <emb_diversity.measures.dist_dispersion.dist_dispersion>` | Sum of all pairwise distances |
| {func}`diameter <emb_diversity.measures.diameter.diameter>` | Largest distance between any two instances |
| {func}`bottleneck <emb_diversity.measures.bottleneck.bottleneck>` | Smallest distance between any two instances |
| {func}`sum_bottleneck <emb_diversity.measures.sum_bottleneck.sum_bottleneck>` | For each point, find its nearest neighbour; sum those distances |
| {func}`sum_diameter <emb_diversity.measures.sum_diameter.sum_diameter>` | For each point, find its farthest neighbour; sum those distances |
| {func}`chamfer_dist <emb_diversity.measures.chamfer_dist.chamfer_dist>` | Average nearest-neighbour distance across all datapoints |
| {func}`span_centroid <emb_diversity.measures.span_centroid.span_centroid>` | Distance-based span relative to the centroid (Cox et al., 2021) |
| {func}`span_medoid <emb_diversity.measures.span_medoid.span_medoid>` | Distance-based span relative to the medoid (Cox et al., 2021) |

## Entropy-based

| Function | Description |
|---|---|
| {func}`vendi_score <emb_diversity.measures.vendi_score.vendi_score>` | Vendi Score -- effective number of distinct items (Friedman & Dieng, 2023) |
| {func}`renyi_entropy <emb_diversity.measures.renyi_entropy.renyi_entropy>` | Renyi Kernel Entropy (RKE) / Matrix-based Renyi entropy |
| {func}`bins_entropy <emb_diversity.measures.bins_entropy.bins_entropy>` | Shannon entropy over a 2D UMAP/PCA-projected histogram |
| {func}`graph_entropy <emb_diversity.measures.graph_entropy.graph_entropy>` | Entropy of the degree distribution of a nearest-neighbour graph |

## Geometry-based

| Function | Description |
|---|---|
| {func}`convex_hull_volume_2d <emb_diversity.measures.convex_hull_volume_2d.convex_hull_volume_2d>` | Area of the convex hull after UMAP-projecting embeddings to 2D |
| {func}`hamdiv <emb_diversity.measures.hamdiv.hamdiv>` | Length of the shortest Hamiltonian circuit through all points |
| {func}`mst_dispersion <emb_diversity.measures.mst_dispersion.mst_dispersion>` | Total edge weight of the minimum spanning tree |
| {func}`radius <emb_diversity.measures.radius.radius>` | Geometric mean of per-dimension standard deviations |

## Distribution-based

| Function | Description |
|---|---|
| {func}`energy <emb_diversity.measures.energy.energy>` | Energy distance between the dataset and a reference distribution |
| {func}`cluster_inertia <emb_diversity.measures.cluster_inertia.cluster_inertia>` | Inertia -- sum of squared distances to cluster centres |
| {func}`dcscore <emb_diversity.measures.dcscore.dcscore>` | DC Score based on self-similarity with softmax normalisation |
| {func}`log_determinant <emb_diversity.measures.log_determinant.log_determinant>` | Log-determinant of the kernel matrix (LDD) -- **default measure** |
