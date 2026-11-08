"""File, process and network isolation for agent code, via a Python audit hook (PEP 578).

The first p1 run showed why ``open`` in the agent's builtins is not enough: the agent listed the repo
with ``os.walk``, read the eval code, blog drafts and ``runs/ledger.csv``, and ran ``git log`` through
``subprocess``. An audit hook sees every file open, directory listing, process spawn and socket
connect made by any code in the process, including imports and C extensions that go through Python.
Once installed it cannot be removed.

The hook is armed only while agent-triggered work runs (``Jail.armed()``), so model loading, setup
code, evaluation and weight saving are unaffected. While armed:

- reads are allowed under the Python installation, installed packages, the model cache, OS paths the
  runtime needs (``/usr``, ``/proc``, ``/dev``...), the workspace and the tool library;
- writes are allowed only in the workspace, the tool library and the temp directory;
- dataset caches, the repository (except the ``wsp`` harness and tool code) and the home directory are
  refused, and so are processes, ``chdir`` and network connections.

This stops an agent that is merely curious. It is not a jail against deliberately hostile code (for
example, ``gc`` can reach the armed flag); every transcript is also audited (``wsp.report.audit``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import sysconfig
import tempfile
from collections.abc import Iterator
from pathlib import Path

DENY_ALWAYS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
               "os.forkpty", "pty.spawn", "os.chdir", "os.fchdir", "socket.connect", "socket.bind",
               "os.kill", "os.chroot", "shutil.rmtree")
PATH_EVENTS = {"open", "os.listdir", "os.scandir", "glob.glob", "os.remove", "os.rename", "os.rmdir",
               "os.mkdir", "os.symlink", "os.link", "os.truncate", "os.chmod", "os.utime",
               "shutil.copyfile", "shutil.move", "shutil.copytree"}
WRITE_EVENTS = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.symlink", "os.link", "os.truncate",
                "os.chmod", "os.utime", "shutil.copyfile", "shutil.move", "shutil.copytree"}
DENY_PARTS = ("datasets--", f"{os.sep}datasets{os.sep}", ".keys", "organisms", "ledger.csv",
              "transcript.jsonl")
WSP_DENY = ("eval", "tasks", "run.py", "rsi.py", "report.py", "export.py", "organisms.py")


class JailViolation(PermissionError):
    pass


def _norm(p: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(p))))


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


class Jail:
    def __init__(self, workspace: str | None = None, library_dir: str | None = None) -> None:
        import wsp

        self.workspace = workspace or tempfile.mkdtemp(prefix="wsp_agent_")
        self.wsp_dir = _norm(Path(wsp.__file__).parent)
        self.wsp_deny = [_norm(Path(self.wsp_dir, d)) for d in WSP_DENY]
        tmp = _norm(tempfile.gettempdir())
        write = [_norm(self.workspace), tmp] + ([_norm(library_dir)] if library_dir else [])
        if os.name != "nt":
            write.append("/dev")
        read = set(write) | {self.wsp_dir}
        for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sysconfig.get_paths().values()):
            if p:
                read.add(_norm(p))
        # Installed packages only. Other sys.path entries (the script's directory, the cwd, a repo root
        # added by an editable install or PYTHONPATH) would open the home directory or the repo.
        for p in sys.path:
            if p and Path(p).is_dir() and any(s in p for s in ("site-packages", "dist-packages")):
                read.add(_norm(p))
        for name in ("interp_engine", "transformers", "torch", "torch_tpu", "safetensors", "numpy"):
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, "__file__", None):
                read.add(_norm(Path(mod.__file__).parent))
        hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        read.add(_norm(os.environ.get("HF_HUB_CACHE") or os.path.join(hf, "hub")))
        read.add(_norm(os.path.join(hf, "modules")))  # transformers' dynamic modules; not the token file
        cache = _norm(os.path.join(os.path.expanduser("~"), ".cache"))
        for sub in ("torch", "torch_tpu", "jax", "libtpu", "triton"):  # compile caches
            read.add(_norm(os.path.join(cache, sub)))
            write.append(_norm(os.path.join(cache, sub)))
        if os.name != "nt":
            read |= {"/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", "/opt", "/bin", "/run"}
        self.read, self.write = sorted(read), write
        self._armed = [False]
        self.violations: list[str] = []
        sys.addaudithook(self._hook)

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        prev_cwd = os.getcwd()
        os.chdir(self.workspace)  # before arming: chdir is refused while armed
        self._armed[0] = True
        try:
            yield
        finally:
            self._armed[0] = False
            os.chdir(prev_cwd)

    def _deny(self, msg: str) -> None:
        self.violations.append(msg)
        raise JailViolation(f"refused by sandbox: {msg}. Agent code may only read the model through the "
                            f"loaded objects and write files in the working directory.")

    def _hook(self, event: str, args: tuple) -> None:
        if not self._armed[0]:
            return
        if event.startswith(DENY_ALWAYS):
            self._deny(f"{event} is not available")
        if event not in PATH_EVENTS or not args:
            return
        target = args[0]
        if target is None or isinstance(target, int):
            return  # fd-based or cwd listing of the workspace
        try:
            path = _norm(target.decode() if isinstance(target, bytes) else target)
        except (TypeError, ValueError):
            return
        writing = event in WRITE_EVENTS or (event == "open" and len(args) > 1 and isinstance(args[1], str)
                                            and any(c in args[1] for c in "wax+"))
        if event == "open" and len(args) > 2 and isinstance(args[2], int):
            writing = writing or bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if any(part in path for part in DENY_PARTS) or _under(path, self.wsp_deny):
            self._deny(f"access to {target!r}")
        if writing and not _under(path, self.write):
            self._deny(f"writing {target!r}")
        if not _under(path, self.read):
            self._deny(f"access to {target!r}")
