# AppWorld Challenge Harness

Install and prepare AppWorld:

```bash
python3.11 -m pip install appworld
python3.11 -m appworld.cli install
python3.11 -m appworld.cli download data
```

The harness defaults to `test_challenge`. Set `APPWORLD_DATASET=test_normal` to run the normal split or set `APPWORLD_ROOT` if data is not stored in the repository root.

Each tool call is executed as Python code through `AppWorld.execute()`. Scoring uses AppWorld's state-based `world.evaluate()` and returns binary task success.
