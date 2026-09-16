from pathlib import Path
from typing import Dict, List, Tuple
import sys

def _human(nbytes: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"

def clean_data_dirs(
    root: str | Path = "SoC_proxy_TSA/data",
    remove_empty_dirs: bool = False,
    preview_examples: int = 5,
) -> Dict[str, Dict[str, int]]:
    """
    Clears files in:
      - <root>/calliope_models   (keeps *reference.netcdf)
      - <root>/cluster_maps
      - <root>/parameters
      - <root>/timeseries        (keeps time_varying_parameters.csv)
    Shows a confirmation with counts and sizes before deleting.

    Returns a summary dict with per-folder counts/sizes actually deleted.
    """
    root = Path(root)
    targets = {
        "calliope_models": root / "calliope_models",
        "cluster_maps":    root / "cluster_maps",
        "parameters":      root / "parameters",
        "timeseries":      root / "timeseries",
    }

    # Build deletion plan
    delete_plan: Dict[str, List[Path]] = {k: [] for k in targets}
    kept_plan: Dict[str, List[Path]] = {k: [] for k in targets}
    total_bytes = 0
    per_dir_bytes: Dict[str, int] = {k: 0 for k in targets}

    for name, d in targets.items():
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            keep = False
            if name == "timeseries" and p.name.lower().startswith("time_varying_parameters"):
                keep = True
            elif name == "timeseries" and p.name.lower().startswith("synthetic"):
                keep = True
            elif name == "calliope_models" and p.name.lower().endswith("reference.nc"):
                keep = True
            elif name == "calliope_models" and p.name.lower().startswith("standard"):
                keep = True
            

            if keep:
                kept_plan[name].append(p)
                continue
            delete_plan[name].append(p)
            try:
                per_dir_bytes[name] += p.stat().st_size
                total_bytes += p.stat().st_size
            except OSError:
                # if we can't stat, still list it for deletion
                pass

    # Tally files
    per_dir_counts = {k: len(v) for k, v in delete_plan.items()}
    total_files = sum(per_dir_counts.values())

    # Preview
    print("\n>>> Cleanup plan:")
    for name in targets:
        d = targets[name]
        exists = " (missing)" if not d.exists() else ""
        print(f" - {name}{exists}: delete {per_dir_counts[name]} files "
              f"({_human(per_dir_bytes[name])}), keep {len(kept_plan[name])}")
        if preview_examples and delete_plan[name]:
            print("    e.g.:")
            for p in delete_plan[name][:preview_examples]:
                print(f"      • {p.relative_to(root)}")
    print(f"\nTOTAL to delete: {total_files} files, {_human(total_bytes)}")
    print("Type 'yes' to proceed with deletion, anything else to abort.")
    ans = input("> ").strip().lower()
    if ans != "yes":
        print("Aborted. No files were deleted.")
        return {"deleted": {"total_files": 0, "total_bytes": 0}}

    # Execute deletion
    deleted = 0
    deleted_bytes = 0
    for name, files in delete_plan.items():
        for p in files:
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            try:
                p.unlink()
                deleted += 1
                deleted_bytes += sz
            except Exception as e:
                print(f" ! Failed to delete {p}: {e}", file=sys.stderr)

        # Optionally remove any empty subdirectories afterwards
        if remove_empty_dirs and targets[name].exists():
            # walk bottom-up so children get removed first
            for sub in sorted(targets[name].rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if sub.is_dir():
                    try:
                        next(sub.iterdir())
                    except StopIteration:
                        try:
                            sub.rmdir()
                        except Exception:
                            pass

    print(f"Deleted {deleted} files, {_human(deleted_bytes)}.")
    return {
        "deleted": {
            "total_files": deleted,
            "total_bytes": deleted_bytes
        },
        "per_dir": {
            name: {"files": per_dir_counts[name], "bytes": per_dir_bytes[name]}
            for name in targets
        }
    }

# Example usage:
# clean_data_dirs("SoC_proxy_TSA/data")
clean_data_dirs()