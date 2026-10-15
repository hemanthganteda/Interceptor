#!/bin/bash
# Copy the Bullet airframe into a PX4-Autopilot tree, register it, and rebuild SITL.
#   px4/install_airframe.sh [~/PX4-Autopilot]
set -euo pipefail
PX4_DIR=${1:-$HOME/PX4-Autopilot}
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes
for f in "$HERE"/airframes/*; do
    name=$(basename "$f")
    cp "$f" "$DST/$name"
    grep -q "^\s*$name\s*$" "$DST/CMakeLists.txt" || sed -i "/^\s*10016_none_iris\s*$/a\\	$name" "$DST/CMakeLists.txt"
    echo "installed $name"
done
make -C "$PX4_DIR" px4_sitl_default
