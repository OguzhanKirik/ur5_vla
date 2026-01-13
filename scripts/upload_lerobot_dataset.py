#!/usr/bin/env python3
"""
Upload a LeRobot dataset (data/ + videos/ + meta/) to the Hugging Face Hub.

Example:
  python upload_lerobot_dataset.py \
    --local-path ./datasets/lerobot/local/ur5_smolvla_grasp \
    --repo-id YOURNAME/ur5_smolvla_grasp \
    --public

Note: The --local-path should point to the folder that CONTAINS data/, videos/, meta/
      (the repo_id folder itself, not its parent)

Optional:
  export HF_TOKEN=hf_xxx   # or pass --hf-token
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def require_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {name} -> {path}")

def find_any_files(path: Path, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if any(path.rglob(pat)):
            return True
    return False

def validate_lerobot_layout(local_path: Path) -> None:
    """Basic sanity checks for the LeRobot dataset folder."""
    require_dir(local_path, "local-path")
    require_dir(local_path / "data", "data/")
    require_dir(local_path / "videos", "videos/")
    require_dir(local_path / "meta", "meta/")

    info_json = local_path / "meta" / "info.json"
    stats_json = local_path / "meta" / "stats.json"
    tasks_parquet = local_path / "meta" / "tasks.parquet"
    episodes_dir = local_path / "meta" / "episodes"

    if not info_json.exists():
        raise FileNotFoundError(f"Missing meta/info.json: {info_json}")
    if not stats_json.exists():
        eprint(f"Warning: meta/stats.json not found at {stats_json} (some pipelines can regenerate it).")
    if not tasks_parquet.exists():
        eprint(f"Warning: meta/tasks.parquet not found at {tasks_parquet} (may be optional depending on setup).")
    if not episodes_dir.exists():
        eprint(f"Warning: meta/episodes/ not found at {episodes_dir} (may be optional depending on setup).")

    # Check for parquet episodes
    has_parquet = find_any_files(local_path / "data", ["*.parquet"])
    if not has_parquet:
        raise FileNotFoundError("No .parquet files found under data/. Did you collect any episodes?")

    # Check for videos in the correct subdirectories
    video_dirs = [
        local_path / "videos" / "observation.images.top",
        local_path / "videos" / "observation.images.wrist"
    ]
    
    has_top_videos = False
    has_wrist_videos = False
    
    if video_dirs[0].exists():
        has_top_videos = find_any_files(video_dirs[0], ["*.mp4", "*.webm", "*.mkv"])
    if video_dirs[1].exists():
        has_wrist_videos = find_any_files(video_dirs[1], ["*.mp4", "*.webm", "*.mkv"])
    
    has_mp4 = find_any_files(local_path / "videos", ["*.mp4", "*.webm", "*.mkv"])
    if not has_mp4:
        eprint("Warning: No video files found under videos/. If you expect videos, check your collection output.")
    else:
        if has_top_videos:
            eprint("✓ Found top camera videos")
        if has_wrist_videos:
            eprint("✓ Found wrist camera videos")

def hf_login_if_needed(hf_token: str | None) -> None:
    """
    If token is provided, configure it for huggingface_hub.
    If not provided, rely on prior `huggingface-cli login`.
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a LeRobot dataset to Hugging Face Hub.")
    parser.add_argument("--local-path", required=True, help="Path to the local dataset folder (contains data/, videos/, meta/).")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id: USERNAME/DATASET_NAME")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (default is private).")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token (or set env HF_TOKEN).")
    parser.add_argument("--skip-load-test", action="store_true", help="Skip quick local load test.")
    args = parser.parse_args()

    local_path = Path(args.local_path).expanduser().resolve()

    try:
        print(f"Validating LeRobot dataset layout at: {local_path}")
        validate_lerobot_layout(local_path)
        print("✅ Layout looks OK.")

        hf_login_if_needed(args.hf_token)

        # Import here so validation errors show before import issues
        from huggingface_hub import upload_folder, create_repo

        if not args.skip_load_test:
            print("Running quick local load test...")
            try:
                # For local datasets, just check if we can read the meta files
                import json
                info_path = local_path / "meta" / "info.json"
                with open(info_path, 'r') as f:
                    info = json.load(f)
                num_episodes = info.get('num_episodes', '?')
                print(f"✅ Local load test passed. Found {num_episodes} episodes.")
            except Exception as e:
                print(f"⚠️  Local load test failed: {e}")
                print("Continuing with upload anyway...")

        print(f"Uploading to Hugging Face Hub: {args.repo_id}")
        
        try:
            # Try to create the repository (will fail gracefully if it exists)
            print("Creating repository (if needed)...")
            create_repo(
                repo_id=args.repo_id,
                repo_type="dataset",
                private=not args.public,
                exist_ok=True,  # Don't fail if it already exists
            )
            print(f"✅ Repository ready: {args.repo_id}")
        except Exception as e:
            print(f"⚠️  Repository check: {e}")
        
        # Upload the folder using HF Hub
        print("Uploading files (this may take a while for large datasets)...")
        upload_folder(
            folder_path=str(local_path),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.hf_token,
            commit_message="Upload LeRobot dataset",
            ignore_patterns=["*.git*", ".git*"],
        )

        visibility = "public" if args.public else "private"
        print(f"✅ Upload complete. Repo should be {visibility}: {args.repo_id}")
        print("Next: open the dataset page → check Files and README, then test loading from hub.")
        return 0

    except Exception as ex:
        eprint("\n❌ Upload failed:")
        eprint(str(ex))
        eprint("\nTips:")
        eprint("- Ensure you ran `huggingface-cli login` or pass --hf-token / set HF_TOKEN.")
        eprint("- Ensure the local path points to the folder that directly contains data/, videos/, meta/.")
        eprint("- Ensure lerobot is installed in this environment.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

