# Progress Spinner

`emb-diversity` shows a progress spinner while an embedding model is downloaded
and prepared — the first call that needs a model can take a while and print a
lot of HuggingFace output, which the spinner replaces. Control it with the
`EMB_DIVERSITY_PROGRESS` environment variable:

- By default the spinner appears only in interactive sessions (a terminal or a
  notebook) and stays silent in scripts, pipes, and CI.
- Set `EMB_DIVERSITY_PROGRESS` to `1`, `true`, `yes`, or `on` to always show it,
  or `0`, `false`, `no`, or `off` to never show it.

Set it before the first call that loads a model. From the shell:

```bash
export EMB_DIVERSITY_PROGRESS=0
```

From Python or a notebook cell:

```python
import os
os.environ["EMB_DIVERSITY_PROGRESS"] = "1"
```
