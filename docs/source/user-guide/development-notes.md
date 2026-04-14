# Development Notes

Internal notes and design decisions for contributors.

## Text input handling

All measure functions accept both raw text and pre-computed embeddings.
This is implemented via the `@accepts_text` decorator in `_accepts_text.py`,
which checks if the first element of the input is a string and, if so,
embeds it before calling the measure function.

The decorator avoids repeating the same type-checking logic in all 20
measure files. An alternative approach would be to add an explicit
`isinstance` check at the top of each function, which is the pattern
used by scikit-learn (see e.g.
[check_array](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py)
for how they validate and convert input types). If the decorator becomes
hard to maintain or debug, switching to explicit checks per function is
a reasonable alternative.

## TODOs

- **TODO:** Check that the core measures set (`CORE_MEASURES` in `_registry.py`) makes sense with Tao.
