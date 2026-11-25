import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from flowrec.mmap_scratch import stage_mmap_file


def test_parallel_staging_reuses_one_complete_copy_and_keeps_older_generation(tmp_path):
    source = tmp_path / "inputs.pt"
    scratch = tmp_path / "scratch"
    contents = b"a" * 100_000
    source.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as pool:
        staged = list(pool.map(lambda _: stage_mmap_file(source, scratch, expected_sha256=digest), range(8)))
    first = staged[0]
    assert set(staged) == {first} and first.read_bytes() == contents
    assert not list(scratch.rglob(".copying"))

    source.write_bytes(b"new contents")
    second = stage_mmap_file(source, scratch)
    assert second != first and second.read_bytes() == b"new contents"
    assert first.read_bytes() == contents


def test_bad_prepared_checksum_never_publishes_staged_input(tmp_path):
    source = tmp_path / "inputs.pt"
    source.write_bytes(b"wrong contents")
    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        stage_mmap_file(source, scratch, expected_sha256="a" * 64)
    assert not list(scratch.rglob("inputs.pt"))
    assert not list(scratch.rglob(".copying"))
