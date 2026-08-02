#!/bin/bash

set -eu

# The script uses paths relative to the project directory, change there before
# doing anything else.
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

if (( $# == 1 )); then
    compute="$1"
else
    compute=CPU
fi
discretization_ratio=0.5
resolution=1

for method in modified original; do
    for dirac in {1..4}; do
        echo "$method method, $compute compute, dirac $dirac"
        ./build/sim_NSE/sim_IBM2 --compute "$compute" --method "$method" --dirac "$dirac" --discretization-ratio "$discretization_ratio" --resolution "$resolution" --spheres 1 --final-time 0.0 --mtx-output
    done
done

# Compare the generated matrices against the baselines. The validator checks
# dimensions, sparsity pattern, and values for every generated file.
python3 tests/regression/check-d3q27-ibm-results.py
