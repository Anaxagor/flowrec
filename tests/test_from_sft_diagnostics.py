"""Preparation diagnostics must not crash or retain the running data graph."""
import faulthandler
import gc
import importlib.util
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "exp8_diagnostics_test", ROOT / "scripts/exp8-from-sft/finetune_from_sft.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
JOB = {"source_id": "gpt2_yambda500_trial4_sampled50"}


def args(interval):
    return SimpleNamespace(traceback_interval=interval, phase="prepare", prepare_kind="cpu")


class NotifyingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.written = threading.Event()

    def write(self, value):
        result = super().write(value)
        if "[exp8 diagnostic stack]" in value:
            self.written.set()
        return result


@pytest.fixture(autouse=True)
def forbid_native_watchdog(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("Native faulthandler watchdog must not be used")
    monkeypatch.setattr(faulthandler, "dump_traceback_later", fail)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", fail)


def test_disabled_periodic_diagnostics_still_reports_stages_without_a_thread(monkeypatch, capsys):
    def fail(*a, **kw):
        raise AssertionError("Disabled diagnostics started a thread")
    monkeypatch.setattr(threading, "Thread", fail)
    with runner.execution_diagnostics(args(0), JOB) as stage:
        stage("prepare_cpu")
    output = capsys.readouterr()
    record = json.loads(output.out)
    assert record["stage"] == "prepare_cpu"
    assert record["source_id"] == JOB["source_id"]
    assert record["type"] == "exp8_stage"
    assert output.err == ""


@pytest.mark.parametrize("raises", [False, True])
def test_periodic_snapshot_captures_runner_and_stops_on_exit(monkeypatch, raises):
    stream = NotifyingStream()
    monkeypatch.setattr(runner.sys, "stderr", stream)
    def preparation_frame_for_snapshot():
        assert stream.written.wait(3), "No Python stack snapshot was emitted"
        if raises:
            raise ValueError("original preparation error")
    def execute():
        with runner.execution_diagnostics(args(0.01), JOB):
            preparation_frame_for_snapshot()
    if raises:
        with pytest.raises(ValueError, match="original preparation error"):
            execute()
    else:
        execute()
    assert "preparation_frame_for_snapshot" in stream.getvalue()
    assert "Timeout (" not in stream.getvalue()
    assert not any(t.name == "exp8-diagnostics" for t in threading.enumerate())


def test_snapshot_does_not_keep_preparation_locals_alive(monkeypatch):
    stream = NotifyingStream()
    monkeypatch.setattr(runner.sys, "stderr", stream)
    class DataGraph:
        pass
    def preparation():
        graph = DataGraph()
        ref = weakref.ref(graph)
        assert stream.written.wait(3)
        return ref
    with runner.execution_diagnostics(args(0.01), JOB):
        ref = preparation()
        gc.collect()
        assert ref() is None


def test_failed_diagnostic_output_does_not_fail_computation(monkeypatch):
    attempted = threading.Event()
    class BrokenStream:
        def write(self, value):
            attempted.set()
            raise OSError("closed diagnostic log")
    monkeypatch.setattr(runner.sys, "stderr", BrokenStream())
    with runner.execution_diagnostics(args(0.01), JOB):
        assert attempted.wait(3)
        result = sum(range(100))
    assert result == 4950
    assert not any(t.name == "exp8-diagnostics" for t in threading.enumerate())
