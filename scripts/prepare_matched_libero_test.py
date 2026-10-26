import copy
import json

import pytest

from . import prepare_matched_libero as prepare


def make_metadata(root):
    meta = root / "meta"
    meta.mkdir(parents=True)
    records = [
        {"episode_index": i, "scene_id": "scene_0", "scene_hash": "hash0", "bddl_file": f"task{i}.bddl"}
        for i in range(11)
    ]
    info = {
        "codebase_version": "v3.0", "total_episodes": 11, "total_frames": 22,
        "features": {
            **{key: {"dtype": "image", "shape": [256, 256, 3]} for key in prepare.CAMERAS},
            "state": {"dtype": "float32", "shape": [8]},
            "actions": {"dtype": "float32", "shape": [7]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "matched_episodes.json").write_text(json.dumps({"episodes": records}))
    (meta / "matched_demos.hdf5").write_bytes(b"archive")
    (meta / "image_convention.json").write_text(json.dumps({
        "raw_simulator_to_dataset": "rotate_180", "frame_skip": 1,
    }))
    return records


def test_matched_bundle_validation(tmp_path):
    make_metadata(tmp_path)
    assert prepare.validate_source(tmp_path, 1)["total_episodes"] == 11
    with pytest.raises(ValueError, match="1650"):
        prepare.validate_source(tmp_path)


@pytest.mark.parametrize("fault", ["duplicate_episode", "duplicate_task", "different_scene", "different_hash"])
def test_rejects_broken_matched_bundles(tmp_path, fault):
    records = copy.deepcopy(make_metadata(tmp_path))
    key, value = {
        "duplicate_episode": ("episode_index", 0),
        "duplicate_task": ("bddl_file", "task0.bddl"),
        "different_scene": ("scene_id", "scene_1"),
        "different_hash": ("scene_hash", "hash1"),
    }[fault]
    records[1][key] = value
    (tmp_path / "meta/matched_episodes.json").write_text(json.dumps({"episodes": records}))
    with pytest.raises(ValueError):
        prepare.validate_source(tmp_path, 1)


def test_rejects_old_nine_task_dataset(tmp_path):
    records = make_metadata(tmp_path)[:9]
    meta = tmp_path / "meta"
    info = json.loads((meta / "info.json").read_text())
    info["total_episodes"] = 9
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "matched_episodes.json").write_text(json.dumps({"episodes": records}))
    with pytest.raises(ValueError, match="11 matched episodes"):
        prepare.validate_source(tmp_path, 1)
