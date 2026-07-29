#!/bin/bash

set -u

# End-to-end ParaView visualization test for the sim_AMR VTKHDF output:
# runs tests/test_amr_paraview_e2e.py under pvpython against
# $SIM_RESULTS_DIR/output_amr_0000.vtkhdf, regenerating the results with
# sim_AMR --resolution 1 first if the file is missing.
#
# Exits 0 on success, 1 on failure, 77 when pvpython is not installed (skip).
# Paths are relative to the project directory, change there before doing
# anything else (same convention as tests/run-amr-tests.sh).
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir" || exit 1

resultsDir="${SIM_RESULTS_DIR:-results_sim_AMR_res01_np001}"
vtkhdf="$resultsDir/output_amr_0000.vtkhdf"
outdir="${PV_E2E_OUTDIR:-/tmp/opencode/pv_e2e}"
logDir="/tmp/opencode"

if ! command -v pvpython >/dev/null 2>&1; then
    echo "[ SKIP ] test_amr_paraview_e2e: pvpython not found on PATH (needs ParaView >= 6.0)"
    exit 77
fi

if [[ ! -f "$vtkhdf" ]]; then
    echo "=== $vtkhdf missing, regenerating with sim_AMR --resolution 1 ==="
    if [[ -d "$resultsDir" ]]; then
        backup="${resultsDir}_bkp_$(date +%Y%m%d_%H%M%S)"
        echo "backing up stale $resultsDir -> $backup"
        mv "$resultsDir" "$backup"
    fi
    simLog="$logDir/sim_amr_regen.log"
    simStatus=0
    ./build/sim_AMR/sim_AMR --resolution 1 >"$simLog" 2>&1 || simStatus=$?
    if (( simStatus != 0 )); then
        echo "[ FAIL ] test_amr_paraview_e2e: sim_AMR exited with $simStatus (see $simLog)"
        exit 1
    fi
    if ! grep -q "physFinalTime reached" "$simLog" && ! grep -q "physFinalTime reached" "$resultsDir/log_main_rank000" 2>/dev/null; then
        echo "[ FAIL ] test_amr_paraview_e2e: sim_AMR finished without 'physFinalTime reached' (see $simLog)"
        exit 1
    fi
    if [[ ! -f "$vtkhdf" ]]; then
        echo "[ FAIL ] test_amr_paraview_e2e: sim_AMR did not produce $vtkhdf (see $simLog)"
        exit 1
    fi
else
    echo "=== reusing existing $vtkhdf ==="
fi

mkdir -p "$outdir" "$logDir"
log="$logDir/test_amr_paraview_e2e.log"
echo "=== running test_amr_paraview_e2e (input: $vtkhdf, outdir: $outdir) ==="
status=0
pvpython tests/test_amr_paraview_e2e.py --input "$vtkhdf" --outdir "$outdir" >"$log" 2>&1 || status=$?

# pvpython prints UCX/HDF5-diagnostic/render-pipeline noise even on success;
# show only the test's own lines (full log kept at $log)
grep -E "^(PASS|FAIL|SKIP|RESULT)" "$log"

if (( status == 0 )); then
    echo "[ PASS ] test_amr_paraview_e2e"
else
    echo "[ FAIL ] test_amr_paraview_e2e (pvpython exited $status; full log: $log)"
fi
exit "$status"
