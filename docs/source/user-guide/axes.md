# Diversity Axes

A **diversity axis** defines *what kind* of diversity you want to measure by mapping to a specific embedding model.
Different embedding models capture different aspects of text, so the same set of texts can have high semantic diversity but low stylistic diversity (or vice versa).

Most built-in axes embed **text** (a list of strings). A few embed a different
modality instead — the `speaker` axis below embeds **audio** (a list of paths
to audio files). The axis you pass decides how raw input is interpreted; the
measure functions themselves are unchanged either way.

## Built-in Axes

### semantic (default)

Measures meaning-based diversity. Texts about different topics will score high; paraphrases will score low.

- **Default model:** `all-mpnet-base-v2`
- **Alternatives:** `all-MiniLM-L6-v2` (faster, smaller)

```python
from emb_diversity import log_determinant

# These are equivalent (semantic is the default)
log_determinant(texts)
log_determinant(texts, diversity_axis="semantic")
```

### style

Measures writing style diversity. Texts written in different styles (formal vs. informal, simple vs. complex) will score high.

- **Default model:** `AnnaWegmann/Style-Embedding`
- **Alternatives:** `StyleDistance/styledistance`, `rrivera1849/LUAR-MUD`, `AIDA-UPM/star`

```python
log_determinant(texts, diversity_axis="style")
```

### speaker

Measures speaker diversity from audio. Pass a list of **paths to audio files**
(not text) — each is embedded with a speaker-discriminative model where the
same speaker's utterances embed close together and different speakers embed
far apart. Requires the `audio` extra (`pip install emb-diversity[audio]`).

- **Default model:** `speechbrain/spkrec-ecapa-voxceleb`

```python
audio_paths = ["alice_1.wav", "alice_2.wav", "bob_1.wav", "carol_1.wav"]
log_determinant(audio_paths, diversity_axis="speaker")
```

## Using a Custom Model

You can bypass the axis system entirely by passing an embedding model directly:

```python
log_determinant(texts, embedding_model="all-MiniLM-L6-v2")
log_determinant(texts, embedding_model="all-mpnet-base-v2")
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
from emb_diversity import axes

for axis in axes.list_all():
    print(f"{axis.name}: default={axis.default_model}")
    if axis.alternative_models:
        print(f"  alternatives: {axis.alternative_models}")
```
