from __future__ import annotations

import argparse
import importlib
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["core", "gpu", "kaggle"], default="gpu")
    args = parser.parse_args()
    required = ["yaml", "pyarrow"] + (["torch", "transformers", "peft", "accelerate", "huggingface_hub", "bm25s", "pyvi"] if args.profile in {"gpu", "kaggle"} else [])
    result = {name: None for name in required}
    for name in required:
        try:
            module = importlib.import_module(name)
            result[name] = getattr(module, "__version__", "imported")
        except Exception as exc:
            result[name] = f"ERROR: {exc!r}"
    print(json.dumps(result, indent=2))
    return 0 if all(not str(value).startswith("ERROR:") for value in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
