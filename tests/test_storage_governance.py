from datetime import datetime, timedelta
import os

from src.services import storage_governance


def test_governance_removes_only_disposable_files(monkeypatch, tmp_path):
    storage = tmp_path / 'storage'
    cache = storage / 'cache'
    learning = storage / 'jingcai' / 'learning'
    cache.mkdir(parents=True)
    learning.mkdir(parents=True)
    disposable = cache / 'response.json'
    protected = learning / 'settled_predictions.csv'
    disposable.write_text('{}', encoding='utf-8')
    protected.write_text('important', encoding='utf-8')
    old = (datetime.now() - timedelta(days=3)).timestamp()
    os.utime(disposable, (old, old))
    monkeypatch.setattr(storage_governance, 'STORAGE_ROOT', storage)

    report = storage_governance.run_storage_governance(dry_run=False)

    assert report['removed_files'] == 1
    assert not disposable.exists()
    assert protected.exists()


def test_governance_dry_run_does_not_delete(monkeypatch, tmp_path):
    storage = tmp_path / 'storage'
    cache = storage / 'tmp'
    cache.mkdir(parents=True)
    disposable = cache / 'stale.part'
    disposable.write_text('partial', encoding='utf-8')
    old = (datetime.now() - timedelta(days=3)).timestamp()
    os.utime(disposable, (old, old))
    monkeypatch.setattr(storage_governance, 'STORAGE_ROOT', storage)

    report = storage_governance.run_storage_governance(dry_run=True)

    assert report['candidate_files'] == 1
    assert report['removed_files'] == 0
    assert disposable.exists()
