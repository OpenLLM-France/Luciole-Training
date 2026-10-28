#!/usr/bin/env python3
"""Probe the judge through its SSH forward using the pipeline's bearer key."""
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import JUDGE_API_KEY  # noqa: E402


def ready(url):
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"} if JUDGE_API_KEY else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


if __name__ == "__main__":
    sys.exit(0 if ready(sys.argv[1]) else 1)
