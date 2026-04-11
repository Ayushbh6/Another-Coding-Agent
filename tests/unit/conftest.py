"""
Shared fixtures for unit tests.

`repo` fixture: a fresh temporary directory used as a fake repo root.
All tool functions accept repo_root as a string — tests pass str(repo) everywhere.

Git is NOT initialised here — the sandbox blocks git. Tools that call git
(get_repo_summary) already handle its absence gracefully and return
"(git not available)", which tests assert accordingly.
"""

import pytest


@pytest.fixture()
def repo(tmp_path):
    """Return a fresh empty directory to use as a fake repo root."""
    return tmp_path
