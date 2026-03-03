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
  To add a new measure: add a new row with the function name (backtick-wrapped) and a short description.
-->

### Distance-based

| Function | Description |
|---|---|
| `mean_pairwise_distance` | Average pairwise distance between all datapoints |
| `distance_dispersion` | Sum of all pairwise distances |
| `diameter` | Largest distance between any two instances |
| `bottleneck` | Smallest distance between any two instances |
| `sum_diameter` | For each point, find its farthest neighbour; sum those distances |
| `chamfer_distance_diversity` | For each point, take the minimum distance to any other; sum these |
| `span_with_centroid` | Distance-based span relative to the centroid (Cox et al., 2021) |
| `span_with_medoid` | Distance-based span relative to the medoid (Cox et al., 2021) |

### Entropy-based

| Function | Description |
|---|---|
| `vendi_score_diversity` | Vendi Score — effective number of distinct items (Friedman & Dieng, 2023) |
| `renyi_kernel_entropy` | Rényi Kernel Entropy (RKE) / Matrix-based Rényi entropy |
| `bins_based_entropy_pca` | Shannon entropy over a 2D PCA-projected histogram |
| `graph_entropy` | Entropy of the degree distribution of a nearest-neighbour graph |

### Geometry-based

| Function | Description |
|---|---|
| `convex_hull_volume` | Volume (or area in 2D) of the convex hull of the embeddings |
| `hamdiv` | Length of the shortest Hamiltonian circuit through all points |
| `mst_dispersion` | Total edge weight of the minimum spanning tree |
| `radius_diversity` | Geometric mean of per-dimension standard deviations |

### Distribution-based

| Function | Description |
|---|---|
| `energy` | Energy distance between the dataset and a reference distribution |
| `cluster_inertia_diversity` | Inertia — sum of squared distances to cluster centres |
| `dcscore` | DC Score based on self-similarity with softmax normalisation |
| `log_determinant_diversity` | Log-determinant of the kernel matrix (LDD) |

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
