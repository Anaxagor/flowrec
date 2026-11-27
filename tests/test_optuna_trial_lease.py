import fcntl
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import optuna

from scripts.optuna_trial_lease import claim_trial, journal_storage


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


def test_four_workers_reclaim_interrupted_trial_and_finish_shared_budget(tmp_path):
    journal = tmp_path / "optuna_journal.log"
    study = optuna.create_study(storage=journal_storage(journal, resumable=True), study_name="search")
    with claim_trial(study, tmp_path, 60) as interrupted:
        assert interrupted.number == 0
        interrupted.report(0.2, step=1)

    start = Barrier(4)
    def worker(_):
        local = optuna.load_study(storage=journal_storage(journal, resumable=True), study_name="search")
        finished = []
        while True:
            with claim_trial(local, tmp_path, 60) as trial:
                if trial is None:
                    return finished
                if not finished:
                    start.wait(timeout=20)  # All four leases must coexist.
                local.tell(trial, 0.5)
                finished.append(trial.number)

    with ThreadPoolExecutor(max_workers=4) as workers:
        done = list(workers.map(worker, range(4)))
    assert sorted(n for numbers in done for n in numbers) == list(range(60))
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert study.trials[0].intermediate_values == {1: 0.2}
