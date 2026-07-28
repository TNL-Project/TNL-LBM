#!/bin/bash

set -u

# Builds and runs the AMR unit tests (tests/test_amr_coupling.cu and
# tests/test_amr_subcycling.cu, each compiled for both the A-B and A-A
# streaming patterns).
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
)

echo "Building AMR test targets: ${targets[*]}"
if ! cmake --build build --target "${targets[@]}"; then
    echo "BUILD FAILED"
    exit 1
fi

status=0
passed=0
for target in "${targets[@]}"; do
    echo "=== running $target ==="
    if ./build/tests/"$target"; then
        echo "PASS: $target"
        passed=$((passed + 1))
    else
        echo "FAIL: $target"
        status=1
    fi
    echo
done

echo "AMR test summary: $passed/${#targets[@]} targets passed"
if (( status == 0 )); then
    echo "All AMR tests passed."
else
    echo "Some AMR tests FAILED."
fi
exit "$status"
