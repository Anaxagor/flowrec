import fcntl
import signal
import subprocess
import sys

import optuna

from scripts.optuna_trial_lease import journal_storage


def test_journal_lock_survives_killed_writer_and_serial_reopen(tmp_path):
    journal = tmp_path / "optuna_journal.log"
    worker = """
import os
import signal
import sys
from pathlib import Path
import optuna
from scripts.optuna_trial_lease import journal_storage

storage = journal_storage(Path(sys.argv[1]), resumable=True)
optuna.create_study(study_name="search", storage=storage)
storage._backend._lock.acquire()
os.kill(os.getpid(), signal.SIGKILL)
"""
    result = subprocess.run([sys.executable, "-c", worker, str(journal)], timeout=20)
    assert result.returncode == -signal.SIGKILL
    marker = tmp_path / "optuna_journal.log.flock"
    with marker.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    storage = journal_storage(journal, resumable=False)
    assert storage._backend._lock.path == marker
    study = optuna.load_study(study_name="search", storage=storage)
    trial = study.ask()
    study.tell(trial, 1.0)
    assert study.best_value == 1.0


def test_existing_journal_keeps_its_original_lock_protocol(tmp_path):
    journal = tmp_path / "optuna_journal.log"
    original = journal_storage(journal, resumable=False)
    optuna.create_study(study_name="existing", storage=original)
    reopened = journal_storage(journal, resumable=True)
    assert type(reopened._backend._lock) is type(original._backend._lock)
    assert not (tmp_path / "optuna_journal.log.flock").exists()
