#!/usr/bin/env python3
from pathlib import Path
import runpy

source = Path(__file__).with_name("apply_sagemath_capability.py")
text = source.read_text(encoding="utf-8")
text = text.replace("module=r'''", 'module=r"""', 1)
marker = "\n'''\n(C/\"sagemath_operations.py\").write_text(module,encoding=\"utf-8\")"
replacement = "\n\"\"\"\n(C/\"sagemath_operations.py\").write_text(module,encoding=\"utf-8\")"
if marker not in text:
    raise SystemExit("unable to locate SageMath module closing marker")
text = text.replace(marker, replacement, 1)
fixed = source.with_name("apply_sagemath_capability.fixed.py")
fixed.write_text(text, encoding="utf-8")
runpy.run_path(str(fixed), run_name="__main__")
