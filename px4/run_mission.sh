#!/bin/bash
# Run sim_px4.py, retrying when PX4 SITL's interval timer stalls at arming (sim_px4.py exits 3).
#   px4/run_mission.sh [sim_px4.py args...]
cd "$(dirname "$0")/.." || exit 1
for attempt in 1 2 3 4 5; do
    /opt/IsaacSim/python.sh sim_px4.py "$@"
    rc=$?
    [ $rc -ne 3 ] && exit $rc
    echo "[run_mission] PX4 airspeed_selector stalled; retrying (attempt $((attempt + 1)))"
done
exit 3
