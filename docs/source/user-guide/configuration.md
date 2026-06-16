# Configuration

This section covers how to tune `emb-diversity`'s runtime behaviour — what it
caches and how it reports progress.

```{toctree}
:maxdepth: 1

cache
progress
```

## Memory use

The command-line interface loads all input texts into memory at once. This is
fine for small- to medium-sized datasets but will not scale to very large files.
