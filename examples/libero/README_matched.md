# Pi0.5 LoRA on matched LIBERO Goal demos

The `pi05_matched_lora` JAX config starts from
`gs://openpi-assets/checkpoints/pi05_libero/params` and trains all 1,650 demos
from `libero_goal_matched_150_multi_targets` (150 shared configurations across
tasks 12, 17, 54–57, and 77–81). The dataset must come
from the parent project's matched collector and camera converter, including
`meta/matched_episodes.json`, `meta/matched_demos.hdf5`, and
`meta/image_convention.json`.

Gemma language and action-expert backbone weights are frozen; their attention
and feedforward LoRA adapters train at built-in ranks 16 and 32, respectively.
The SigLIP vision encoder and action projections train in full. Thus the VLM
adapts through language adapters and vision weight updates. EMA is disabled.
The default batch size is 32, with 20,000 steps, 500 warmup steps, a cosine
learning rate from 2.5e-5 to 2.5e-6, and saves every 2,000 steps.

## Prepare the dataset

Render the 11-task collected HDF5 run using the standard schema first, from the
parent checkout with its LangGap environment activated:

```bash
python langgap/convert/convert_matched_libero_goal.py \
  --run-dir /absolute/path/to/matched-run \
  --repo-id local/libero_goal_matched_150_multi_targets --standard-libero
```

The new 11-task batch job runs this export and the v2.1 preparation below after
collection automatically. The older nine-task job keeps its existing schema.

OpenPi pins LeRobot v2.1, while the matched collector uses LeRobot v3. Keep the
two Python environments separate. From the parent project, run
`./setup_env.sh openpi` if needed. Then, from `openpi/`:

```bash
../.venv-openpi/bin/python scripts/prepare_matched_libero.py \
  /absolute/path/to/matched-run/lerobot \
  --output /absolute/path/to/matched-run/lerobot-openpi-v21 \
  --source-python "$PWD/../.venv-langgap/bin/python"
```

This decodes with the source environment and writes with OpenPi's pinned
LeRobot. It uses one temporary episode at a time and writes a new dataset with
embedded images, preserving frame/episode order, task strings, FPS, state, and actions.
Images are stored losslessly; this conversion does not encode videos. No images are
rotated again, no actions are converted to deltas again, and no frames are
trimmed. Matched metadata is copied unchanged. Existing outputs are refused;
the final output appears only after successful conversion and validation.

If the matched dataset is already v2.1, skip this step and pass it directly to
the launcher. These exports use the standard LIBERO columns: `image`,
`wrist_image`, `state`, and `actions`. Both cameras are embedded image columns
in episode Parquet files, matching the standard Hugging Face LIBERO layout.
The pinned writer emits LeRobot v2.1 (the reference dataset is v2.0); the feature
schema is the same. Training uses `LeRobotLiberoDataConfig` directly. Prompts
come from lowercase task metadata.

## Train

```bash
# Short smoke run; inspect logged camera views and loss before the full run.
EXP_NAME=matched_smoke bash scripts/train_matched_libero.sh \
  /absolute/path/to/matched-run/lerobot-openpi-v21 \
  --batch-size 2 --num-train-steps 10 --save-interval 5 --num-workers 0

EXP_NAME=matched_vlm_lora bash scripts/train_matched_libero.sh \
  /absolute/path/to/matched-run/lerobot-openpi-v21

# Resume the same experiment; reuse statistics only for the same dataset.
EXP_NAME=matched_vlm_lora SKIP_NORM_STATS=1 \
  bash scripts/train_matched_libero.sh \
  /absolute/path/to/matched-run/lerobot-openpi-v21 --resume
```

`OPENPI_PYTHON` overrides the training Python executable. The launcher validates
the matched bundle metadata, registers a local dataset symlink under
`HF_LEROBOT_HOME/local/libero_goal_matched_150_multi_targets`, computes normalization statistics,
and invokes `scripts/train.py`. It refuses to repoint an existing dataset cache.
For standalone checkouts, use `.venv/bin/python` and provide the v3 environment
path explicitly during preparation. The first run may download model/tokenizer
assets and requires access to their hosts. Training logs to W&B by default;
pass `--no-wandb-enabled` to disable it.

Checkpoints are under
`checkpoints/pi05_matched_lora/<experiment>/<step>`; statistics are stored in
`assets/pi05_matched_lora/local/libero_goal_matched_150_multi_targets`. Do not run normalization
for different datasets concurrently with this same config/asset directory.

Serve a saved checkpoint with:

```bash
../.venv-openpi/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_matched_lora \
  --policy.dir checkpoints/pi05_matched_lora/matched_vlm_lora/19999
```

At inference, send the standard OpenPi LIBERO keys (`observation/image`,
`observation/wrist_image`, `observation/state`, `prompt`), with the same camera
orientation and 8-D EEF axis-angle state used for training. Use the bundled
matched snapshots for matched-scene evaluation. Such evaluation measures
performance on training configurations; a held-out evaluation must split by
scene ID across all eleven tasks.
