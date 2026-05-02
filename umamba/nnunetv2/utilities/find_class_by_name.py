import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            try:
                m = importlib.import_module(current_module + "." + modname)
            except Exception as e:
                # Skip trainer modules that fail to import (e.g. optional CUDA-only deps
                # like mamba_ssm missing on CPU/Turing boxes). One broken sibling shouldn't
                # block discovery of an unrelated trainer.
                print(f"[recursive_find_python_class] skipping {current_module}.{modname}: "
                      f"{type(e).__name__}: {e}")
                continue
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr