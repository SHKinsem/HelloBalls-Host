#!/usr/bin/env bash
# ROS Humble's setup.bash reads optional environment variables that may be
# unset. Explicitly disable nounset first because an invoking shell can export
# SHELLOPTS and make a new Bash process inherit `set -u`.
set +u
set -eo pipefail

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

# CycloneDDS discovers the camera endpoints on this host but has repeatedly
# failed to deliver the full-size image samples. Keep bag recording on the
# same Fast DDS implementation used by the camera and VINS processes.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

set -u

mkdir -p "${BAG_ROOT}"

echo "Recording rosbag2 to ${OUTPUT_DIR}"
printf 'Topics:\n'
printf '  %s\n' "${TOPICS[@]}"

exec ros2 bag record -o "${OUTPUT_DIR}" "${TOPICS[@]}"
