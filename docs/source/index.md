# Diversity Measurement Documentation

```{include} ../../README.md
:start-after: <!-- docs-intro-start -->
:end-before: <!-- docs-intro-end -->
```

## Installation

```{include} ../../README.md
:start-after: <!-- docs-install-start -->
:end-before: <!-- docs-install-end -->
```

## Usage

```{include} ../../README.md
:start-after: <!-- docs-usage-start -->
:end-before: <!-- docs-usage-end -->
```

## Available Measures

<!--
  To update measure groupings, edit the tables below.
  Each entry corresponds to a function in measure_diversity.measure.
  To move a measure to a different group: cut its row and paste it into the correct table.
  To add a new measure: add a new row using the {func} syntax below and a short description.
  Format: | {func}`display_name <measure_diversity.measure.function_name>` | description |
-->

### Distance-based

| Function | Description |
|---|---|
| {func}`mean_pairwise_distance <measure_diversity.measure.mean_pairwise_distance>` | Average pairwise distance between all datapoints |
| {func}`distance_dispersion <measure_diversity.measure.distance_dispersion>` | Sum of all pairwise distances |
| {func}`diameter <measure_diversity.measure.diameter>` | Largest distance between any two instances |
| {func}`bottleneck <measure_diversity.measure.bottleneck>` | Smallest distance between any two instances |
| {func}`sum_diameter <measure_diversity.measure.sum_diameter>` | For each point, find its farthest neighbour; sum those distances |
| {func}`chamfer_distance_diversity <measure_diversity.measure.chamfer_distance_diversity>` | For each point, take the minimum distance to any other; sum these |
| {func}`span_with_centroid <measure_diversity.measure.span_with_centroid>` | Distance-based span relative to the centroid (Cox et al., 2021) |
| {func}`span_with_medoid <measure_diversity.measure.span_with_medoid>` | Distance-based span relative to the medoid (Cox et al., 2021) |

### Entropy-based

| Function | Description |
|---|---|
| {func}`vendi_score_diversity <measure_diversity.measure.vendi_score_diversity>` | Vendi Score — effective number of distinct items (Friedman & Dieng, 2023) |
| {func}`renyi_kernel_entropy <measure_diversity.measure.renyi_kernel_entropy>` | Rényi Kernel Entropy (RKE) / Matrix-based Rényi entropy |
| {func}`bins_based_entropy_pca <measure_diversity.measure.bins_based_entropy_pca>` | Shannon entropy over a 2D PCA-projected histogram |
| {func}`graph_entropy <measure_diversity.measure.graph_entropy>` | Entropy of the degree distribution of a nearest-neighbour graph |

### Geometry-based

| Function | Description |
|---|---|
| {func}`convex_hull_volume <measure_diversity.measure.convex_hull_volume>` | Volume (or area in 2D) of the convex hull of the embeddings |
| {func}`hamdiv <measure_diversity.measure.hamdiv>` | Length of the shortest Hamiltonian circuit through all points |
| {func}`mst_dispersion <measure_diversity.measure.mst_dispersion>` | Total edge weight of the minimum spanning tree |
| {func}`radius_diversity <measure_diversity.measure.radius_diversity>` | Geometric mean of per-dimension standard deviations |

### Distribution-based

| Function | Description |
|---|---|
| {func}`energy <measure_diversity.measure.energy>` | Energy distance between the dataset and a reference distribution |
| {func}`cluster_inertia_diversity <measure_diversity.measure.cluster_inertia_diversity>` | Inertia — sum of squared distances to cluster centres |
| {func}`dcscore <measure_diversity.measure.dcscore>` | DC Score based on self-similarity with softmax normalisation |
| {func}`log_determinant_diversity <measure_diversity.measure.log_determinant_diversity>` | Log-determinant of the kernel matrix (LDD) |

## API Reference

```{toctree}
:maxdepth: 2
:caption: Contents

measure_diversity
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
