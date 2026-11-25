"""Coordinate same-study Optuna workers on a shared filesystem with flock."""

import fcntl
from contextlib import ExitStack, contextmanager
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


class JournalFlock:
    """JournalFileBackend lock released by the kernel after process death."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self):
        self.handle = self.path.open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except BaseException:
            self.handle.close()
            raise
        return True

    def release(self):
        self.handle.close()


def journal_storage(path: Path, *, resumable: bool):
    """Choose the journal's locking scheme exactly once, before its first write.

    Existing unmarked journals keep Optuna's original symlink locking. New
    resumable journals record their flock choice so later serial/reporting
    invocations also use it. All writers to a new study must use this factory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix(path.suffix + ".flock")
    with path.with_suffix(path.suffix + ".init.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not path.exists() and resumable:
            marker.touch(exist_ok=True)
        lock = JournalFlock(marker) if marker.exists() else None
        backend = JournalFileBackend(str(path), lock_obj=lock)
    return JournalStorage(backend)


@contextmanager
def allocation_lock(dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / ".allocation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def worker_mode_lock(dataset_dir: Path, *, parallel: bool):
    """Prevent concurrent serial and parallel tuners, including legacy TIGER."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / ".worker.lock").open("a") as handle:
        mode = fcntl.LOCK_SH if parallel else fcntl.LOCK_EX
        try:
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Conflicting tuning worker already owns {dataset_dir}") from exc
        yield


@contextmanager
def claim_trial(study, dataset_dir: Path, n_trials: int, *, count_failures: bool = False):
    """Yield an abandoned or newly allocated trial, or None when all slots are taken.

    Hold the trial lock through study.tell(). The OS also releases it when a
    worker dies. Other workers can then recover that exact RUNNING trial.
    """
    trial = None
    with ExitStack() as lease:
        with allocation_lock(dataset_dir):
            for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                handle = (dataset_dir / f".trial_{frozen.number}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                # A finishing worker may have told the study since our snapshot.
                if study._storage.get_trial(frozen._trial_id).state != TrialState.RUNNING:
                    handle.close()
                    continue
                lease.enter_context(handle)
                trial = optuna.Trial(study, frozen._trial_id)
                break

            if trial is None:
                allocated = (TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED)
                if count_failures:
                    allocated += (TrialState.FAIL,)
                if len(study.get_trials(deepcopy=False, states=allocated)) < n_trials:
                    trial = study.ask()
                    handle = lease.enter_context((dataset_dir / f".trial_{trial.number}.lock").open("a"))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield trial
