
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from foliage_bones import build_rig
from foliage_debug_drawer import _stream_settings
from foliage_streams import compute_all_streams

PROJECT_ROOT = Path(__file__).parent.parent.parent
WIND_NODE = (
    PROJECT_ROOT / "python_classes" / "nodes" / "export_nodes" / "foliage_wind_node.py"
)
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "foliage_streams"

RIG_ONLY_KEYS = {"build_bone_rig", "segments_per_chain"}


def authored_settings() -> dict:
    module = ast.parse(WIND_NODE.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_FOLIAGE_SETTINGS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_FOLIAGE_SETTINGS not found in foliage_wind_node.py")


def get_settings_keys() -> set:
    module = ast.parse(WIND_NODE.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "get_settings":
            for statement in ast.walk(node):
                if isinstance(statement, ast.Dict):
                    return {
                        key.value
                        for key in statement.keys
                        if isinstance(key, ast.Constant)
                    }
    raise AssertionError("get_settings not found in foliage_wind_node.py")


def synthetic_tree(per_chain: int = 6) -> dict:
    positions, stem_ids, extents, radii, pivots, directions, depths = [], [], [], [], [], [], []

    for stem_id, depth, base, direction, extent, radius in (
        (0, 0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 4.0, 0.5),
        (1, 1, np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]), 2.0, 0.2),
    ):
        for step in range(per_chain):
            t = step / max(per_chain - 1, 1)
            positions.append(base + direction * extent * t)
            stem_ids.append(float(stem_id))
            extents.append(extent)
            radii.append(radius)
            pivots.append(base)
            directions.append(direction)
            depths.append(float(depth))

    return {
        "vertex_positions": np.array(positions),
        "stem_ids": np.array(stem_ids),
        "branch_extents": np.array(extents),
        "radii": np.array(radii),
        "pivot_positions": np.array(pivots),
        "directions": np.array(directions),
        "hierarchy_depths": np.array(depths),
    }


class TestSettingsContract:

    def test_every_authored_setting_reaches_a_consumer(self):
        authored = authored_settings()

        consumers = {
            "compute_all_streams": set(inspect.signature(compute_all_streams).parameters),
            "build_rig": set(inspect.signature(build_rig).parameters),
        }

        tree = synthetic_tree()

        filtered = _stream_settings(authored)
        compute_all_streams(**tree, **filtered)

        rig_settings = {
            key: value for key, value in authored.items() if key in consumers["build_rig"]
        }
        build_rig(**tree, **rig_settings)

        with pytest.raises(TypeError):
            compute_all_streams(**tree, **authored)

        routed = {
            key: sorted(name for name, params in consumers.items() if key in params)
            for key in authored
        }

        violations = [
            f"{key} is accepted by no consumer"
            for key, names in sorted(routed.items())
            if not names and key not in RIG_ONLY_KEYS
        ]

        lines = [
            "# authored foliage settings and the consumers that accept them",
            f"# consumers: {', '.join(sorted(consumers))}",
            "",
            "# columns: setting | default | accepted_by",
        ]
        for key in sorted(authored):
            lines.append(f"{key} {authored[key]!r} {', '.join(routed[key]) or 'NONE'}")

        dropped = sorted(set(authored) - set(filtered))
        lines.extend(
            [
                "",
                "# rig-only keys the stream filter removes",
                "  " + (", ".join(dropped) if dropped else "none"),
                "",
                "# violations",
            ]
            + ([f"  {v}" for v in violations] or ["  none"])
        )

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "settings_contract.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        assert not violations, "; ".join(violations)

    def test_defaults_match_get_settings(self):
        assert authored_settings().keys() == get_settings_keys()
