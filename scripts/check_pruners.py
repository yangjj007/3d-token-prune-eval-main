#!/usr/bin/env python3
"""Verify pruner modules import and register correctly (run from eval-main root)."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PRUNER_MODULES: tuple[str, ...] = (
    "eval.baseline",
    "eval.pruners.baseline",
    "eval.proposed.loco3d",
    "eval.proposed.octree_merge",
    "eval.proposed.reconot",
    "eval.proposed.runlength_curve",
)

_BASELINE_SOURCES = frozenset({"otprune", "apet", "divprune", "tome", "fastv_mesh"})


def _source_path(name: str) -> Path:
    if name in _BASELINE_SOURCES:
        return REPO_ROOT / "eval" / "baseline" / f"{name}.py"
    if name in {"no_pruning", "random", "uniform"}:
        return REPO_ROOT / "eval" / "pruners" / "baseline.py"
    return REPO_ROOT / "eval" / "proposed" / f"{name}.py"


def _import_all_pruner_modules() -> dict[str, str]:
    errors: dict[str, str] = {}
    for mod in _PRUNER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            errors[mod] = f"{type(exc).__name__}: {exc}"
    return errors


def _registry_has_lazy_loader() -> bool:
    from eval import pruners as pruners_pkg

    return callable(getattr(pruners_pkg, "ensure_pruners_loaded", None))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pruner registry")
    parser.add_argument(
        "names",
        nargs="*",
        default=["reconot", "loco3d", "otprune"],
        help="Pruner names to verify (default: reconot loco3d otprune)",
    )
    args = parser.parse_args()

    init_py = REPO_ROOT / "eval" / "pruners" / "__init__.py"
    print(f"eval_main_root={REPO_ROOT}")
    print(f"pruners_init={init_py} exists={init_py.is_file()}")
    print(f"lazy_loader={'yes' if _registry_has_lazy_loader() else 'NO — sync eval/pruners/__init__.py'}")

    import_errors = _import_all_pruner_modules()

    from eval.pruners import PRUNER_REGISTRY, get_pruner_class

    print(f"registered={sorted(PRUNER_REGISTRY.keys())}")

    if import_errors:
        print("\nimport_errors:")
        for mod, err in sorted(import_errors.items()):
            print(f"  {mod}: {err}")

    failed = False
    for name in args.names:
        path = _source_path(name)
        print(f"\n--- {name} ---")
        print(f"source={path} exists={path.is_file()}")
        if not path.is_file():
            print("FAIL: source file missing on disk")
            failed = True
            continue
        try:
            cls = get_pruner_class(name)
            print(f"ok class={cls.__name__}")
        except KeyError as exc:
            print(f"FAIL: {exc}")
            failed = True
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            failed = True

    if failed and not _registry_has_lazy_loader():
        print(
            "\nHint: server eval/pruners/__init__.py is outdated. "
            "Sync these files from your dev machine:\n"
            "  eval-main/eval/pruners/__init__.py\n"
            "  eval-main/eval/run_eval.py\n"
            "  eval-main/eval/proposed/reconot.py"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
