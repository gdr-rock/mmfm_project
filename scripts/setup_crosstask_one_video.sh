#!/usr/bin/env bash
set -euo pipefail

# Downloads CrossTask assets on the remote machine and extracts exactly one video.
# Usage:
#   bash scripts/setup_crosstask_one_video.sh [target_dir]
# Example:
#   bash scripts/setup_crosstask_one_video.sh "$HOME/datasets/crosstask"

TARGET_DIR="${1:-$HOME/datasets/crosstask}"
VIDEOS_DIR="$TARGET_DIR/videos"
RELEASE_ZIP="$TARGET_DIR/crosstask_release.zip"
MISSING_TAR="$TARGET_DIR/missing_videos.tar.gz"
INDEX_FILE="$TARGET_DIR/missing_videos_index.txt"
ONE_PATH_FILE="$TARGET_DIR/one_video_path.txt"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_ID_FILE="$REPO_ROOT/configs/video_ids_example.txt"

mkdir -p "$TARGET_DIR" "$VIDEOS_DIR"

echo "[1/5] Download CrossTask release zip"
wget -c -O "$RELEASE_ZIP" "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"

echo "[2/5] Download replacement videos archive"
wget -c -O "$MISSING_TAR" "https://www.rocq.inria.fr/cluster-willow/dzhukov/missing_videos.tar.gz"

echo "[3/5] Build archive index and choose first video"
tar -tzf "$MISSING_TAR" > "$INDEX_FILE"
awk '/\.(mp4|webm|mkv|avi)$/ {print; exit}' "$INDEX_FILE" > "$ONE_PATH_FILE"
ONE_ARCHIVE_PATH="$(cat "$ONE_PATH_FILE")"

if [[ -z "$ONE_ARCHIVE_PATH" ]]; then
  echo "No video file found inside $MISSING_TAR"
  exit 1
fi

echo "[4/5] Extract exactly one video: $ONE_ARCHIVE_PATH"
tar -xzf "$MISSING_TAR" -C "$VIDEOS_DIR" "$ONE_ARCHIVE_PATH"

LOCAL_VIDEO_PATH="$VIDEOS_DIR/$ONE_ARCHIVE_PATH"
if [[ ! -f "$LOCAL_VIDEO_PATH" ]]; then
  echo "Expected extracted file not found: $LOCAL_VIDEO_PATH"
  exit 1
fi

VIDEO_BASENAME="$(basename "$LOCAL_VIDEO_PATH")"
VIDEO_ID="${VIDEO_BASENAME%.*}"

echo "[5/5] Write single-video include file: $VIDEO_ID_FILE"
printf "# one video id per line (no extension)\n%s\n" "$VIDEO_ID" > "$VIDEO_ID_FILE"

cat <<EOF

Done.
- videos_root for config: $VIDEOS_DIR
- extracted video path: $LOCAL_VIDEO_PATH
- selected video_id: $VIDEO_ID

Now set in configs/subset_example.yaml:
  dataset.videos_root: "$VIDEOS_DIR"
  dataset.subset_size: 1
  dataset.include_video_ids_file: "configs/video_ids_example.txt"
EOF
