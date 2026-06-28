#!/usr/bin/env bash
set -euo pipefail

BAG_ROOT="${BAG_ROOT:-bags}"
BAG_NAME="${BAG_NAME:-sensor_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${BAG_ROOT}/${BAG_NAME}"

TOPICS=(
  /camera/image_raw
  /camera/camera_info
  /imu/data_raw
  /tf
  /tf_static
)

if [[ -n "${EXTRA_TOPICS:-}" ]]; then
  read -r -a EXTRA_TOPIC_ARRAY <<< "${EXTRA_TOPICS}"
  TOPICS+=("${EXTRA_TOPIC_ARRAY[@]}")
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi

mkdir -p "${BAG_ROOT}"

echo "Recording rosbag2 to ${OUTPUT_DIR}"
printf 'Topics:\n'
printf '  %s\n' "${TOPICS[@]}"

exec ros2 bag record -o "${OUTPUT_DIR}" "${TOPICS[@]}"
