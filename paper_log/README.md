# Paper-trading audit log

This directory holds the live state of the forward-paper portfolio, written
exclusively by the `paper-run` job in `.github/workflows/algo.yml`.

- `state.json` — current cash, positions, signal history, fill history, and
  daily mark-to-market equity points. Committed to the repo so the entire
  trail is reproducible from `git log paper_log/`.

The local CLI uses `state/paper_portfolio.json` (gitignored) by default.
The workflow passes `--state paper_log/state.json` to keep operational state
separate from local exploration state.

If you want to start the paper run with an opening manual trade, run locally:

```
PYTHONPATH=src python -m algo.live.cli \
    --state paper_log/state.json confirm \
    --symbol JUNIORBEES --side BUY --qty 12 --price 773.07
```

then commit `paper_log/state.json`. The next workflow run picks up from there.
