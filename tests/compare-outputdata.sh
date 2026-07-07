#!/bin/bash

set -eu

# The script uses paths relative to the project directory, change there before
# doing anything else.
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

# Lattice resolution passed to the test simulation (default: 1).
resolution="${1:-1}"

# BP5 file-based output test (all outputs at once).
python3 ./tests/validate-outputdata.py --engine bp5 --output-kind all --resolution "$resolution"

# SST streaming output test (one output category at a time).
for kind in 3d 3dcut 2d; do
    python3 ./tests/validate-outputdata.py --engine sst --output-kind "$kind" --resolution "$resolution"
done

# Inline/Plugin in-situ output test (single 3D output).
python3 ./tests/validate-outputdata.py --engine inline --output-kind 3d --resolution "$resolution"
