import argparse
import json
import os
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute notebook code cells without Jupyter.")
    parser.add_argument("--notebook", required=True, help="Path to .ipynb file")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after first failing cell")
    args = parser.parse_args()

    notebook_path = os.path.abspath(args.notebook)
    repo_root = os.path.dirname(os.path.dirname(notebook_path))
    os.chdir(repo_root)

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    globals_dict = {"__name__": "__main__"}
    n_ok = 0
    n_fail = 0

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"\n=== Running cell {idx} ===")
        try:
            code = compile(source, f"<notebook-cell-{idx}>", "exec")
            exec(code, globals_dict, globals_dict)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"Cell {idx} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if args.stop_on_error:
                break

    print("\n=== Notebook execution summary ===")
    print(f"Succeeded cells: {n_ok}")
    print(f"Failed cells:    {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
