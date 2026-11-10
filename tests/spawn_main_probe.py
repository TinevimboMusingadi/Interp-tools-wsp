"""Run as ``python -m spawn_main_probe`` (from tests/): a main module that imports the task registry, like
``python -m wsp.run`` does. Prints what agent code can find of the registry inside the sandbox."""

import wsp.tasks.registry  # noqa: F401  - the import spawn would replay in the child
from wsp.harness.sandbox import Sandbox

PROBE = ("import gc, types, sys\n"
         "fns = sorted({o.__name__ for o in gc.get_objects() if isinstance(o, types.FunctionType)"
         " and o.__code__.co_filename.replace(chr(92), '/').endswith('wsp/tasks/registry.py')})\n"
         "mods = sorted(k for k in sys.modules if k.startswith('wsp.tasks') or k in ('wsp.run', '__mp_main__'))\n"
         "print('FOUND', fns, mods)")

if __name__ == "__main__":
    with Sandbox(loader="tiny:tiny_gpt2", loader_kwargs={}, toolset="c0", forward_budget=1) as sb:
        print(sb.call("python", {"code": PROBE}))
