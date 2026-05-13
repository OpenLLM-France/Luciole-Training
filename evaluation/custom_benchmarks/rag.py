"""Aggregator file for `--custom-tasks`.

Lighteval's `--custom-tasks` accepts a single module, so this file re-exports a
combined `TASKS_TABLE` from every sibling benchmark file. Pass this file as
`--custom-tasks evaluation/custom_benchmarks/all_tasks.py` to get every task at
once.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from hotpotqa import TASKS_TABLE as _hotpotqa_tasks  # noqa: E402
from longbench import TASKS_TABLE as _longbench_tasks  # noqa: E402


TASKS_TABLE = [
    *_hotpotqa_tasks,
    *_longbench_tasks,
]
