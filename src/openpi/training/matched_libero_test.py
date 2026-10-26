from unittest import mock

import flax.nnx as nnx
import numpy as np
import pytest

from openpi import transforms
from openpi.training import config


def test_matched_mapping_preserves_cameras_state_actions_and_prompt(tmp_path):
    cfg = config.get_config("pi05_matched_lora")
    # Tokenizer assets are unrelated to the dataset mapping and need not be downloaded.
    with mock.patch.object(config.ModelTransformFactory, "__call__", return_value=transforms.Group()):
        data = cfg.data.create(tmp_path, cfg.model)
    raw = {
        "image": np.full((3, 256, 256), 0.2, dtype=np.float32),
        "wrist_image": np.full((3, 256, 256), 0.8, dtype=np.float32),
        "state": np.arange(8, dtype=np.float32),
        "actions": np.full((10, 7), 0.25, dtype=np.float32),
        "prompt": "put the bowl in the cabinet",
    }
    result = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])(raw)
    assert data.repo_id == "local/libero_goal_matched_150_multi_targets"
    assert data.asset_id == "local/libero_goal_matched_150_multi_targets"
    assert type(cfg.data) is config.LeRobotLiberoDataConfig
    assert data.action_sequence_keys == ("actions",)
    assert data.prompt_from_task
    assert data.use_quantile_norm
    np.testing.assert_array_equal(result["actions"], raw["actions"])
    np.testing.assert_array_equal(result["state"], raw["state"])
    assert result["prompt"] == raw["prompt"]
    assert np.all(result["image"]["base_0_rgb"] == 51)
    assert np.all(result["image"]["left_wrist_0_rgb"] == 204)
    assert not result["image_mask"]["right_wrist_0_rgb"]


@pytest.mark.parametrize(("path", "frozen"), [
    (("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), True),
    (("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), True),
    (("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"), False),
    (("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "lora_b"), False),
    (("PaliGemma", "img", "embedding", "kernel"), False),
    (("action_in_proj", "kernel"), False),
])
def test_matched_freeze_filter(path, frozen):
    cfg = config.get_config("pi05_matched_lora")
    parameter = nnx.Param(np.ones(1))
    assert nnx.filterlib.to_predicate(cfg.freeze_filter)(path, parameter) == frozen
    assert nnx.filterlib.to_predicate(cfg.trainable_filter)(path, parameter) != frozen


def test_matched_rejects_double_delta_transform(tmp_path):
    data = config.LeRobotMatchedLiberoDataConfig(extra_delta_transform=True)
    with pytest.raises(ValueError, match="already deltas"):
        data.create(tmp_path, config.get_config("pi05_matched_lora").model)
