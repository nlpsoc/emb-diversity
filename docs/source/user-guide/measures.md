# Available Measures

embediver provides 20 diversity measures across four categories.
All measures accept either raw text (list of strings) or pre-computed embeddings (array of vectors).

## Distance-based

| Function | Description |
|---|---|
| {func}`mean_pw_dist <embediver.measures.mean_pw_dist.mean_pw_dist>` | Average pairwise distance between all datapoints |
| {func}`dist_dispersion <embediver.measures.dist_dispersion.dist_dispersion>` | Sum of all pairwise distances |
| {func}`diameter <embediver.measures.diameter.diameter>` | Largest distance between any two instances |
| {func}`bottleneck <embediver.measures.bottleneck.bottleneck>` | Smallest distance between any two instances |
| {func}`sum_diameter <embediver.measures.sum_diameter.sum_diameter>` | For each point, find its farthest neighbour; sum those distances |
| {func}`chamfer_dist <embediver.measures.chamfer_dist.chamfer_dist>` | Average nearest-neighbour distance across all datapoints |
| {func}`span_centroid <embediver.measures.span_centroid.span_centroid>` | Distance-based span relative to the centroid (Cox et al., 2021) |
| {func}`span_medoid <embediver.measures.span_medoid.span_medoid>` | Distance-based span relative to the medoid (Cox et al., 2021) |

## Entropy-based

| Function | Description |
|---|---|
| {func}`vendi_score <embediver.measures.vendi_score.vendi_score>` | Vendi Score -- effective number of distinct items (Friedman & Dieng, 2023) |
| {func}`renyi_entropy <embediver.measures.renyi_entropy.renyi_entropy>` | Renyi Kernel Entropy (RKE) / Matrix-based Renyi entropy |
| {func}`bins_entropy <embediver.measures.bins_entropy.bins_entropy>` | Shannon entropy over a 2D UMAP/PCA-projected histogram |
| {func}`graph_entropy <embediver.measures.graph_entropy.graph_entropy>` | Entropy of the degree distribution of a nearest-neighbour graph |

## Geometry-based

| Function | Description |
|---|---|
| {func}`convex_hull_volume <embediver.measures.convex_hull_volume.convex_hull_volume>` | Volume (or area in 2D) of the convex hull of the embeddings |
| {func}`hamdiv <embediver.measures.hamdiv.hamdiv>` | Length of the shortest Hamiltonian circuit through all points |
| {func}`mst_dispersion <embediver.measures.mst_dispersion.mst_dispersion>` | Total edge weight of the minimum spanning tree |
| {func}`radius <embediver.measures.radius.radius>` | Geometric mean of per-dimension standard deviations |

## Distribution-based

| Function | Description |
|---|---|
| {func}`energy <embediver.measures.energy.energy>` | Energy distance between the dataset and a reference distribution |
| {func}`cluster_inertia <embediver.measures.cluster_inertia.cluster_inertia>` | Inertia -- sum of squared distances to cluster centres |
| {func}`dcscore <embediver.measures.dcscore.dcscore>` | DC Score based on self-similarity with softmax normalisation |
| {func}`log_determinant <embediver.measures.log_determinant.log_determinant>` | Log-determinant of the kernel matrix (LDD) -- **default measure** |
