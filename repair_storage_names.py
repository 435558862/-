#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/administrator/ProphitBet/storage").resolve()
EXPECTED_ROOT = Path("/home/administrator/ProphitBet/storage")


def repaired_name(name: str) -> str | None:
    try:
        repaired = name.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return repaired if repaired != name else None


def main() -> None:
    if ROOT != EXPECTED_ROOT or not ROOT.is_dir():
        raise RuntimeError(f"Unexpected storage root: {ROOT}")

    candidates: list[tuple[Path, Path]] = []
    for path in ROOT.rglob("*"):
        fixed = repaired_name(path.name)
        if fixed is not None:
            candidates.append((path, path.with_name(fixed)))

    collisions = [(old, new) for old, new in candidates if new.exists()]
    if collisions:
        details = "\n".join(f"{old} -> {new}" for old, new in collisions)
        raise FileExistsError(f"Refusing to overwrite existing paths:\n{details}")

    # Rename children before their parent directories so every source path
    # remains valid until it is processed.
    candidates.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    for old, new in candidates:
        old.rename(new)
        print(f"{old.relative_to(ROOT)} -> {new.name}")

    print(f"Repaired {len(candidates)} storage path names.")


if __name__ == "__main__":
    main()
