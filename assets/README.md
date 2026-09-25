# Evaluation assets

Both files are bundled so `python -m refun.train` and `python -m refun.predict`
run out of the box with no extra downloads. They affect **scoring only** — the
model never sees them, so they cannot change what it predicts, only how the
prediction is measured.

| File | Size | Used by |
|---|---|---|
| `segmentation.model` | 507 KB | token-level precision / recall / F1 |
| `word_cluster.json` | 423 KB | CWordNet-F1 |

## `segmentation.model`

A SentencePiece model that splits identifiers into subtokens, so that
`write_build_id`, `writeBuildID` and `WriteBuildId` all reduce to the same
token sequence before comparison. Without it, `refun.train` falls back to plain
camelCase/underscore splitting — slightly coarser, and the token-F1 numbers
shift a little, so results computed with and without it are not directly
comparable.

```
write_build_id        ->  ▁write ▁build ▁id
HandleBuildIDSection  ->  ▁handle build id section
fputs_unlocked        ->  ▁f put s ▁unlock ed
```

## `word_cluster.json`

CodeWordNet synonym clusters: 18,379 entries mapping a word to the cluster IDs
it belongs to. Two tokens count as matching if they share a cluster, so a
prediction of `fetch_entry` against a gold `get_entry` is credited rather than
scored as a miss. This is what separates CWordNet-F1 from plain token-F1;
without the file the two metrics are identical.

```json
{"get": [1, 3, 5257, 4746, 11, ...], "find": [1, 1100, 49, ...], ...}
```

## Overriding

Point the environment variables (or the CLI flags) elsewhere to use your own:

```bash
export REFUN_SP_MODEL=/path/to/segmentation.model
export REFUN_WORD_CLUSTER=/path/to/word_cluster.json
# or
python -m refun.train ... --sp_model_path ... --word_cluster_path ...
```

A missing asset degrades the corresponding metric with a printed warning rather
than aborting the run. `--require_assets` makes absence a hard error instead.
