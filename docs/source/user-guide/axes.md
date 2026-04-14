# Diversity Axes

A **diversity axis** defines *what kind* of diversity you want to measure by mapping to a specific embedding model.
Different embedding models capture different aspects of text, so the same set of texts can have high semantic diversity but low stylistic diversity (or vice versa).

## Built-in Axes

### semantic (default)

Measures meaning-based diversity. Texts about different topics will score high; paraphrases will score low.

- **Default model:** `all-mpnet-base-v2`
- **Alternatives:** `all-MiniLM-L6-v2` (faster, smaller)

```python
from embediver import log_determinant

# These are equivalent (semantic is the default)
log_determinant(texts)
log_determinant(texts, diversity_axis="semantic")
```

### style

Measures writing style diversity. Texts written in different styles (formal vs. informal, simple vs. complex) will score high.

- **Default model:** `AnnaWegmann/Style-Embedding`

```python
log_determinant(texts, diversity_axis="style")
```

## Using a Custom Model

You can bypass the axis system entirely by passing an embedding model directly:

```python
log_determinant(texts, embedding_model="Qwen/Qwen3-8B")
log_determinant(texts, embedding_model="all-MiniLM-L6-v2")
```

The `embedding_model` parameter always overrides `diversity_axis` when both are provided.

## Using Alternative Models

Each axis lists alternative models that work well for its purpose.
To use an alternative, pass it via `embedding_model`:

```python
# Use the faster alternative for semantic diversity
log_determinant(texts, embedding_model="all-MiniLM-L6-v2")
```

Discover available alternatives programmatically:

```python
from embediver import list_axes

for axis in list_axes():
    print(f"{axis.name}: default={axis.default_model}")
    if axis.alternative_models:
        print(f"  alternatives: {axis.alternative_models}")
```

## Registering Custom Axes

You can register your own axes for repeated use:

```python
from embediver import register_axis

register_axis(
    "multilingual",
    default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    alternative_models=["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
    description="Cross-lingual semantic diversity",
)

# Now use it like any built-in axis
from embediver import log_determinant
log_determinant(texts, diversity_axis="multilingual")
```

Custom axes are available for the duration of the Python process.
To make them permanent, add the `register_axis` call to your project's setup code or a config module.
