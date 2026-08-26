from src.version import __version__
from scripts.health_check import run


def test_release_version_is_semantic():
    parts = __version__.split('.')
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_health_check_returns_structured_lists():
    passed, failed = run()
    assert isinstance(passed, list)
    assert isinstance(failed, list)
    assert any(item.startswith('产品：') for item in passed)

