"""YAML configuration loader with native PyYAML support and standalone hierarchical fallback."""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def parse_val(v: str) -> Any:
    """Parses primitive scalar/list value from a YAML string."""
    v = v.split("#")[0].strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("none", "null"):
        return None
    if v.startswith("[") and v.endswith("]"):
        items = [parse_val(x.strip()) for x in v[1:-1].split(",") if x.strip()]
        return items
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v.strip("\"'")


def load_yaml(filepath_or_dict: str | Dict[str, Any]) -> Dict[str, Any]:
    """Loads a YAML file into nested dictionaries/lists.

    Uses PyYAML if installed; otherwise falls back to a standalone parser.

    Args:
        filepath_or_dict: Path to .yaml file, or existing dictionary.

    Returns:
        Hierarchical configuration dictionary.
    """
    if isinstance(filepath_or_dict, dict):
        return filepath_or_dict

    if not isinstance(filepath_or_dict, str) or not os.path.exists(filepath_or_dict):
        return {}

    if yaml is not None:
        try:
            with open(filepath_or_dict, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass

    # Standalone hierarchical YAML parser fallback
    with open(filepath_or_dict, "r", encoding="utf-8") as f:
        lines = [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    i = 0
    while i < len(lines):
        raw = lines[i]
        indent = len(raw) - len(raw.lstrip())
        content = raw.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if content.startswith("- "):
            val = parse_val(content[2:])
            if isinstance(parent, list):
                parent.append(val)
            i += 1
        elif ":" in content:
            k, v = content.split(":", 1)
            k = k.strip()
            v = v.split("#")[0].strip()

            if not v:
                # Lookahead to see whether child is list (- ) or dict
                if i + 1 < len(lines):
                    next_raw = lines[i + 1]
                    next_content = next_raw.strip()
                    if next_content.startswith("- "):
                        new_container: Any = []
                    else:
                        new_container = {}
                else:
                    new_container = {}

                if isinstance(parent, dict):
                    parent[k] = new_container
                stack.append((indent, new_container))
                i += 1
            else:
                if isinstance(parent, dict):
                    parent[k] = parse_val(v)
                i += 1
        else:
            i += 1

    return root
