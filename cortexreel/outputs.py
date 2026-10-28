"""Writing under ``out/`` without following links.

``out/`` and every folder inside it must be real folders, not symlinks or
Windows junctions, and an existing output file is unlinked before it is
rewritten, so a planted link can never redirect a write somewhere else.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

OUT_ROOT = Path("out")
# The stat module only defines these on Windows; the values are fixed by the NT API.
_LINK_TAGS = (
    getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
    getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
)


class OutputError(ValueError):
    pass


def is_link(path: str | Path) -> bool:
    """True for a symlink or a Windows junction (``os.path.islink`` misses junctions on 3.11)."""
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISLNK(st.st_mode) or getattr(st, "st_reparse_tag", 0) in _LINK_TAGS


def out_root() -> Path:
    """``./out`` resolved; refuses a symlinked or junctioned ``out``."""
    root = Path.cwd() / OUT_ROOT
    if is_link(root):
        raise OutputError(f"Refusing to write through a link: {root}")
    return root.resolve()


def check_no_links(path: str | Path) -> None:
    """Refuse ``path`` if it, or a folder between it and ``./out``, is a link.

    Paths outside ``./out`` (an explicit ``--out``) are the user's choice and not checked.
    """
    path = Path(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(Path.cwd() / OUT_ROOT))
    chain = (path, *path.parents)
    keys = [os.path.normcase(str(p)) for p in chain]
    if root not in keys:
        return
    for p, key in zip(chain, keys):
        if is_link(p):
            raise OutputError(f"Refusing to write through a link: {p}")
        if key == root:
            return


def rel_path(path: str | Path) -> str:
    """``path`` relative to the CWD for JSON outputs (absolute paths carry the user name).

    Stays absolute when no relative path exists (another drive on Windows).
    """
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path.cwd().resolve())).as_posix()
    except ValueError:
        return str(path)


def same_file(a: str | Path, b: str | Path) -> bool:
    """Same path, or the same file on disk (hardlinks, symlinks, case on Windows)."""
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve():
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist yet
        return False


def prepare_file(path: str | Path) -> Path:
    """Make ``path``'s folder and remove an existing file there, so the write creates a new one."""
    path = Path(path)
    check_no_links(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_no_links(path.parent)  # mkdir may have raced with a planted link
    if is_link(path) or path.is_file():
        path.unlink()
    elif path.exists():
        raise OutputError(f"Output path is not a file: {path}")
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = prepare_file(path)
    path.write_text(text, encoding="utf-8")
    return path
