#!/bin/bash

set -u

# Builds and runs the doctest-based AMR unit-test binaries: the four gate
# suites (coupling, subcycling, vtkhdf_writer, nesting) consolidated into
# tests/unit/test_amr_units_{ab,aa} (compiled once per streaming pattern),
# driven per TEST_SUITE via doctest's --test-suite filter below, followed by
# the two ParaView end-to-end visualization tests
# (tests/test_amr_paraview_e2e.sh and the 3-level nesting arm
# tests/test_amr_paraview_e2e_nesting.sh driven by the dedicated mock
# build/tests/test_amr_nesting_sim, each skipped with exit 77 when pvpython
# is not installed).
#
# The script uses paths relative to the project directory, change there before
# doing anything else (same convention as tests/compare-IBM-matrices.sh).
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

binaries=(
    test_amr_units_ab
    test_amr_units_aa
)

suites=(
    amr_coupling
    amr_subcycling
    amr_vtkhdf_writer
    amr_nesting
)

echo "Building AMR test targets: ${binaries[*]} test_amr_nesting_sim"
if ! cmake --build build --target "${binaries[@]}" test_amr_nesting_sim; then
    echo "BUILD FAILED"
    exit 1
fi

status=0
passed=0
counted=0
for binary in "${binaries[@]}"; do
    for suite in "${suites[@]}"; do
        echo "=== running $binary --test-suite=$suite ==="
        if ./build/tests/"$binary" --test-suite="$suite" --no-colors --no-duration; then
            echo "PASS: $binary/$suite"
            passed=$((passed + 1))
        else
            echo "FAIL: $binary/$suite"
            status=1
        fi
        counted=$((counted + 1))
        echo
    done
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

# the 3-level nesting e2e arm (the plan's commit D): driven by the dedicated
# mock tests/test_amr_nesting_sim built above
echo "=== running test_amr_paraview_e2e_nesting ==="
paraviewStatus=0
bash tests/test_amr_paraview_e2e_nesting.sh || paraviewStatus=$?
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
