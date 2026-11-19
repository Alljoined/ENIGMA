"""Check ENIGMA input paths without importing ML libraries or opening data files."""

import argparse
import json
import string
from pathlib import Path


PATH_FIELDS = (
    "metadata_path", "eeg_trials_train", "eeg_trials_test",
    "features_train", "features_test",
)


def inspect_layout(config, root=Path("."), subjects=None):
    """Return path availability only, never schema or scientific validity."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    configured = config.get("subjects")
    if not isinstance(configured, list) or not configured or any(
        not isinstance(s, str) or not s for s in configured
    ):
        raise ValueError("subjects must be a nonempty list of nonempty strings")
    if len(set(configured)) != len(configured):
        raise ValueError("subjects must not contain duplicates")
    selected = configured if subjects is None else subjects
    if not selected or any(s not in configured for s in selected):
        raise ValueError("selected subjects must be listed in config")
    root = Path(root)
    checked = {}
    for field in PATH_FIELDS:
        template = config.get(field)
        if not isinstance(template, str) or not template:
            raise ValueError(f"{field} must be a nonempty string")
        try:
            fields = list(string.Formatter().parse(template))
        except ValueError as exc:
            raise ValueError(f"invalid template for {field}: {exc}") from exc
        if any(name not in (None, "subject") or spec or conversion
               for _, name, spec, conversion in fields):
            raise ValueError(f"{field} supports only the {{subject}} placeholder")
        for subject in selected:
            path = root / template.format(subject=subject)
            key = str(path)
            if key not in checked:
                checked[key] = {"path": key, "fields": [], "exists": path.is_file()}
            if field not in checked[key]["fields"]:
                checked[key]["fields"].append(field)
    records = list(checked.values())
    missing = [record for record in records if not record["exists"]]
    return {
        "scope": "file availability only; contents, alignment and provenance not validated",
        "subjects": list(selected),
        "checked": len(records),
        "missing": len(missing),
        "files": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Resolve configured relative paths here; default is current directory")
    parser.add_argument("--subjects", nargs="+", help="Configured subject IDs to check")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = inspect_layout(config, args.root, args.subjects)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Input check failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
