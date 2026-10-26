"""Re-export matched LeRobot v3 demos to OpenPi's pinned LeRobot v2.1 format.

Run with OpenPi's Python; --source-python points to the newer LeRobot environment.
Only one episode of decoded frames is staged at a time. No images are rotated,
actions transformed, frames dropped, or data uploaded.
"""

import argparse
from collections import defaultdict
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

CAMERAS = ("image", "wrist_image")
FEATURES = (*CAMERAS, "state", "actions")
MATCHED_TASK_COUNT = 11
SIDECARS = ("matched_episodes.json", "matched_demos.hdf5", "image_convention.json")


def validate_source(root: Path, expected_scenes: int = 150) -> dict:
    """Check complete eleven-task bundles before creating any output."""
    if expected_scenes < 1:
        raise ValueError("expected_scenes must be positive")
    info = json.loads((root / "meta/info.json").read_text())
    for name in SIDECARS:
        if not (root / "meta" / name).is_file():
            raise ValueError(f"Missing matched metadata: {name}")
    convention = json.loads((root / "meta/image_convention.json").read_text())
    if convention.get("raw_simulator_to_dataset") != "rotate_180" or convention.get("frame_skip") != 1:
        raise ValueError("Expected unskipped demos already in HF LIBERO image orientation")
    records = json.loads((root / "meta/matched_episodes.json").read_text())["episodes"]
    if len(records) != expected_scenes * MATCHED_TASK_COUNT or info["total_episodes"] != len(records):
        raise ValueError(f"Expected {expected_scenes * MATCHED_TASK_COUNT} matched episodes")
    if info["total_frames"] < len(records):
        raise ValueError("Expected nonempty matched episodes")
    if sorted(record["episode_index"] for record in records) != list(range(len(records))):
        raise ValueError("Matched episode indices must cover the dataset exactly once")
    scenes = defaultdict(dict)
    for record in records:
        scene = scenes[record["scene_id"]]
        if record["bddl_file"] in scene:
            raise ValueError("Duplicate task in a matched scene")
        scene[record["bddl_file"]] = record["scene_hash"]
    task_sets = {tuple(sorted(scene)) for scene in scenes.values()}
    hashes = {next(iter(scene.values())) for scene in scenes.values()}
    if (
        len(scenes) != expected_scenes
        or len(task_sets) != 1
        or any(len(scene) != MATCHED_TASK_COUNT or len(set(scene.values())) != 1 for scene in scenes.values())
        or len(hashes) != expected_scenes
    ):
        raise ValueError("Expected eleven tasks sharing each of the unique matched configurations")
    for key, shape in [(key, [256, 256, 3]) for key in CAMERAS] + [
        ("state", [8]),
        ("actions", [7]),
    ]:
        feature = info["features"][key]
        if feature["shape"] != shape or (key in CAMERAS and feature["dtype"] != "image"):
            raise ValueError(f"Unexpected matched feature: {key}")
    return info


def export_worker(source: Path, spool: Path) -> None:
    # Keep stdout exclusively for the parent/worker protocol, including during imports.
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = json.loads((source / "meta/info.json").read_text())
        dataset = LeRobotDataset("local/libero_goal_matched_150_multi_targets", root=source, video_backend="pyav")
        offset = 0
        for episode in range(info["total_episodes"]):
            frames = {key: [] for key in FEATURES}
            tasks = []
            while offset < len(dataset):
                row = dataset[offset]
                row_episode = int(row["episode_index"])
                if row_episode != episode:
                    if row_episode != episode + 1:
                        raise ValueError("Source episodes are not contiguous and ordered")
                    break
                for key in FEATURES:
                    value = row[key].cpu().numpy()
                    if key in CAMERAS:
                        value = np.rint(value.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    frames[key].append(value)
                tasks.append(row["task"])
                offset += 1
            if not tasks:
                raise ValueError(f"Empty source episode {episode}")
            path = spool / "episode.npz"
            np.savez(path, **{key: np.stack(value) for key, value in frames.items()}, task=np.asarray(tasks))
            print(json.dumps({"episode": episode, "length": len(tasks)}), file=protocol, flush=True)
            if sys.stdin.readline().strip() != "ok":
                raise RuntimeError("Conversion parent stopped")
        if offset != info["total_frames"]:
            raise ValueError("Source frame count differs from metadata")


def prepare(source: Path, output: Path, source_python: str, expected_scenes: int = 150) -> None:
    info = validate_source(source, expected_scenes)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if info["codebase_version"] != "v3.0":
        raise ValueError("This exporter expects v3.0; v2.1 datasets can be trained directly")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix=".matched-v21-", dir=output.parent) as temporary,
        contextlib.ExitStack() as cleanup,
    ):
        staging = Path(temporary)
        features = {key: dict(info["features"][key]) for key in FEATURES}
        # v3's numeric-axis names may use a dictionary; v2.1 expects a list.
        for feature in features.values():
            feature["shape"] = tuple(feature["shape"])
            if isinstance(feature.get("names"), dict):
                feature["names"] = next(iter(feature["names"].values()))
        dataset = LeRobotDataset.create(
            repo_id="local/libero_goal_matched_150_multi_targets",
            root=staging / "dataset",
            fps=info["fps"],
            features=features,
            robot_type=info.get("robot_type", "panda"),
            use_videos=False,
            image_writer_threads=4,
            video_backend="pyav",
        )
        cleanup.callback(dataset.stop_image_writer)
        command = [source_python, str(Path(__file__).resolve()), str(source), "--export-worker", str(staging)]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
            try:
                for line in worker.stdout:
                    record = json.loads(line)
                    if record["episode"] != dataset.meta.total_episodes:
                        raise ValueError("Exported episodes changed order")
                    with np.load(staging / "episode.npz", allow_pickle=False) as episode:
                        arrays = {key: episode[key] for key in (*FEATURES, "task")}
                    for index in range(record["length"]):
                        dataset.add_frame(
                            {
                                **{key: arrays[key][index] for key in FEATURES},
                                "task": str(arrays["task"][index]),
                            }
                        )
                    del arrays
                    dataset.save_episode()
                    (staging / "episode.npz").unlink()
                    worker.stdin.write("ok\n")
                    worker.stdin.flush()
                    print(f"Saved episode {dataset.meta.total_episodes}/{info['total_episodes']}", flush=True)
                if worker.wait() != 0:
                    raise RuntimeError("Source LeRobot reader failed; see its stderr above")
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
        if (
            dataset.meta.total_episodes != info["total_episodes"]
            or dataset.meta.total_frames != info["total_frames"]
        ):
            raise ValueError("Re-export changed episode/frame counts")
        dataset.stop_image_writer()
        for name in SIDECARS:
            shutil.copy2(source / "meta" / name, staging / "dataset/meta" / name)
        validate_source(staging / "dataset", expected_scenes)
        (staging / "dataset").rename(output)
    print(f"OpenPi dataset ready: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Matched run's lerobot directory")
    parser.add_argument("--output", type=Path, help="New v2.1 output directory; must not exist")
    parser.add_argument("--source-python", help="Python executable with the converter's LeRobot v3 installed")
    parser.add_argument("--expected-scenes", type=int, default=150)
    parser.add_argument("--check-only", action="store_true", help="Validate a prepared v2.1 dataset without writing")
    parser.add_argument("--export-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        info = validate_source(args.source.resolve(), args.expected_scenes)
        if info["codebase_version"] != "v2.1":
            parser.error("Training needs v2.1. Re-export with --output and --source-python first.")
        print(f"Validated {info['total_episodes']} matched episodes, {info['total_frames']} frames")
    elif args.export_worker:
        export_worker(args.source.resolve(), args.export_worker)
    else:
        if args.output is None or args.source_python is None:
            parser.error("--output and --source-python are required")
        prepare(args.source.resolve(), args.output.resolve(), args.source_python, args.expected_scenes)


if __name__ == "__main__":
    main()
