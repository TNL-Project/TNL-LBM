#!/bin/bash

set -u

# Builds and runs the AMR unit tests (tests/test_amr_coupling.cu,
# tests/test_amr_subcycling.cu, tests/test_amr_vtkhdf_writer.cu and
# tests/test_amr_nesting.cu, each compiled for both the A-B and A-A
# streaming patterns), followed by the ParaView end-to-end visualization
# test (tests/test_amr_paraview_e2e.sh, skipped with exit 77 when pvpython
# is not installed).
#
# The script uses paths relative to the project directory, change there before
# doing anything else (same convention as tests/compare-IBM-matrices.sh).
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

targets=(
    test_amr_coupling_ab
    test_amr_coupling_aa
    test_amr_subcycling_ab
    test_amr_subcycling_aa
    test_amr_vtkhdf_writer_ab
    test_amr_vtkhdf_writer_aa
    test_amr_nesting_ab
    test_amr_nesting_aa
)

echo "Building AMR test targets: ${targets[*]}"
if ! cmake --build build --target "${targets[@]}"; then
    echo "BUILD FAILED"
    exit 1
fi

status=0
passed=0
counted=0
for target in "${targets[@]}"; do
    echo "=== running $target ==="
    if ./build/tests/"$target"; then
        echo "PASS: $target"
        passed=$((passed + 1))
    else
        echo "FAIL: $target"
        status=1
    fi
    counted=$((counted + 1))
    echo
done

echo "=== running test_amr_paraview_e2e ==="
paraviewStatus=0
bash tests/test_amr_paraview_e2e.sh || paraviewStatus=$?
if (( paraviewStatus == 0 )); then
    passed=$((passed + 1))
    counted=$((counted + 1))
elif (( paraviewStatus == 77 )); then
    : # pvpython not installed: test skipped, not counted
else
    status=1
    counted=$((counted + 1))
fi
echo

echo "AMR test summary: $passed/$counted targets passed"
if (( status == 0 )); then
    echo "All AMR tests passed."
else
    echo "Some AMR tests FAILED."
fi
exit "$status"
