#!/bin/bash

set -u

# End-to-end ParaView visualization test for the 3-level nesting mock
# VTKHDF output: runs tests/test_amr_paraview_e2e_nesting.py under pvpython
# against $SIM_RESULTS_DIR/output_amr_0000.vtkhdf, regenerating the results
# with the dedicated driver build/tests/test_amr_nesting_sim first if the
# file is missing (sim_AMR/sim_AMR_channel hardcode their region specs, so
# the nesting arm drives the compact mock instead).
#
# Exits 0 on success, 1 on failure, 77 when pvpython is not installed (skip).
# Paths are relative to the project directory, change there before doing
# anything else (same convention as tests/run-amr-tests.sh).
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir" || exit 1

resultsDir="${SIM_RESULTS_DIR:-results_test_amr_nesting_sim_np001}"
vtkhdf="$resultsDir/output_amr_0000.vtkhdf"
outdir="${PV_E2E_OUTDIR:-/tmp/opencode/pv_e2e_nesting}"
logDir="/tmp/opencode"

if ! command -v pvpython >/dev/null 2>&1; then
    echo "[ SKIP ] test_amr_paraview_e2e_nesting: pvpython not found on PATH (needs ParaView >= 6.0)"
    exit 77
fi

if [[ ! -f "$vtkhdf" ]]; then
    echo "=== $vtkhdf missing, regenerating with test_amr_nesting_sim ==="
    if [[ -d "$resultsDir" ]]; then
        backup="${resultsDir}_bkp_$(date +%Y%m%d_%H%M%S)"
        echo "backing up stale $resultsDir -> $backup"
        mv "$resultsDir" "$backup"
    fi
    simLog="$logDir/test_amr_nesting_sim_regen.log"
    simStatus=0
    ./build/tests/test_amr_nesting_sim >"$simLog" 2>&1 || simStatus=$?
    if (( simStatus != 0 )); then
        echo "[ FAIL ] test_amr_paraview_e2e_nesting: test_amr_nesting_sim exited with $simStatus (see $simLog)"
        exit 1
    fi
    if ! grep -q "physFinalTime reached" "$simLog" && ! grep -q "physFinalTime reached" "$resultsDir/log_main_rank000" 2>/dev/null; then
        echo "[ FAIL ] test_amr_paraview_e2e_nesting: test_amr_nesting_sim finished without 'physFinalTime reached' (see $simLog)"
        exit 1
    fi
    if [[ ! -f "$vtkhdf" ]]; then
        echo "[ FAIL ] test_amr_paraview_e2e_nesting: test_amr_nesting_sim did not produce $vtkhdf (see $simLog)"
        exit 1
    fi
else
    echo "=== reusing existing $vtkhdf ==="
fi

mkdir -p "$outdir" "$logDir"
log="$logDir/test_amr_paraview_e2e_nesting.log"
echo "=== running test_amr_paraview_e2e_nesting (input: $vtkhdf, outdir: $outdir) ==="
status=0
pvpython tests/test_amr_paraview_e2e_nesting.py --input "$vtkhdf" --outdir "$outdir" >"$log" 2>&1 || status=$?

# pvpython prints UCX/HDF5-diagnostic/render-pipeline noise even on success;
# show only the test's own lines (full log kept at $log)
grep -E "^(PASS|FAIL|SKIP|RESULT)" "$log"

if (( status == 0 )); then
    echo "[ PASS ] test_amr_paraview_e2e_nesting"
else
    echo "[ FAIL ] test_amr_paraview_e2e_nesting (pvpython exited $status; full log: $log)"
fi
exit "$status"
