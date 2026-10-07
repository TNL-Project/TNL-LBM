#!/usr/bin/env -S just --working-directory . --justfile

set dotenv-load

# Flags for tools that do not respect the FORCE_COLOR environment variable

color := if env("FORCE_COLOR", "") == "1" { "--color" } else { "" }
color_always := if env("FORCE_COLOR", "") == "1" { "--color always" } else { "" }

# NOTE: The configure recipe is intended to be replaceable - users can run cmake manually
#       with their desired options and then use other recipes like `build` or `check`.

# Configures the project for building with cmake using options from the .env file and environment variables
configure:
    #!/usr/bin/env bash
    set -euo pipefail

    # Collect all variables that start with CMAKE_ or TNL_LBM_ (Note that `env` sees
    # only exported variables, but `set` is a built-in command that sees all
    # variables defined in the .env file.)
    declare -a cmake_options
    for var in $(set | grep -E '^CMAKE_|^TNL_LBM_'); do
        cmake_options+=(-D "$var")
    done

    just _ensure-command cmake
    cmake -B "$BUILD_DIR" -S . "${cmake_options[@]}"

# Lists available build targets
list-build-targets:
    just _ensure-cmake-configured
    cmake --build "$BUILD_DIR" --target help 2>/dev/null | grep ": phony" | grep -v "/" | sed 's/: phony//'

# Builds the project using cmake (this is the default recipe)
[default]
build +targets="all":
    just _ensure-cmake-configured
    cmake --build "$BUILD_DIR" --target {{ targets }}

# Runs all tests using pytest (extra args are forwarded to pytest, e.g. just test tests/unit -x)
test *args: (build "all")
    just _ensure-command pytest
    pytest {{ args }}

# Cleans the build directory
clean:
    rm -frv "$BUILD_DIR"

# Runs all checks
check: check-typos check-format check-recipes check-python check-clangd-tidy

# Checks for common spelling mistakes using typos
check-typos:
    just _ensure-command typos
    typos {{ color_always }} --sort

# Checks the code formatting using clang-format, gersemi, tombi, and ruff
check-format:
    just --unstable --fmt --check
    just _ensure-command clang-format
    # Note that find -exec always exits with 0 exit code, whereas xargs runs
    # clang-format only once and preserves its exit code.
    find ./include/ ./sim_NSE/ ./sim_NSE_ADE/ ./sim_2D/ ./sim_adjoint/ ./sim_non-Newtonian/ ./tests/ ./pytnl_lbm/ \
         \( -name '*.h' \
         -o -name '*.hpp' \
         -o -name '*.cpp' \
         -o -name '*.cu' \) \
         -print0 | xargs -0 clang-format --dry-run -Werror --style file
    just _ensure-command gersemi
    gersemi {{ color }} --diff --check .
    just _ensure-command tombi
    tombi format --check --diff .
    just _ensure-command ruff
    ruff format --diff

# Reformats supported files using clang-format, gersemi, tombi, and ruff
format:
    just --unstable --fmt
    just _ensure-command clang-format
    # Note that find -exec always exits with 0 exit code, whereas xargs runs
    # clang-format only once and preserves its exit code.
    find ./include/ ./sim_NSE/ ./sim_NSE_ADE/ ./sim_2D/ ./sim_adjoint/ ./sim_non-Newtonian/ ./tests/ ./pytnl_lbm/ \
         \( -name '*.h' \
         -o -name '*.hpp' \
         -o -name '*.cpp' \
         -o -name '*.cu' \) \
         -print0 | xargs -0 clang-format -i --style file
    just _ensure-command gersemi
    gersemi {{ color }} --in-place .
    just _ensure-command tombi
    tombi format .
    just _ensure-command ruff
    ruff format .

# Checks justfile recipe for shell issues using shellcheck
_check-recipe recipe:
    just _ensure-command grep shellcheck
    just -vv -n {{ recipe }} 2>&1 | grep -iv '===> Running recipe' | shellcheck -

# Checks all justfile recipes with inline bash for shell issues using shellcheck
check-recipes:
    just _check-recipe '_ensure-command command'
    just _check-recipe '_ensure-compile-commands-json'
    just _check-recipe 'configure'
    just _check-recipe 'check-clangd-tidy'

# Checks Python code quality using ruff
check-ruff:
    just _ensure-command ruff
    ruff check

# Checks Python type annotations using pyright
check-typing:
    just _ensure-command pyright
    pyright

# Checks Python code quality using ruff and pyright
check-python: check-ruff check-typing

# Lints all library headers and simulation sources with clangd-tidy (via compile_commands.json)
check-clangd-tidy:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command clangd-tidy
    just _ensure-compile-commands-json

    root="$(pwd -P)/"
    mapfile -t headers < <(python3 -c "import json; print(*sorted({e['file'] for e in json.load(open('$BUILD_DIR/compile_commands.json')) if '/include/' in e['file']}), sep='\n')")
    mapfile -t sources < <(python3 -c "import json; print(*sorted({e['file'] for e in json.load(open('$BUILD_DIR/compile_commands.json')) if e['file'].endswith(('.cu', '.cpp')) and e['file'].startswith('$root') and not e['file'].startswith('$root' + 'build/')}), sep='\n')")
    # headers are cheap, but every swallowed source costs up to ~1 GB of resident preamble in the single clangd process,
    # so the source pass caps the number of parallel slots to bound memory usage
    jobs_sources=$(( $(nproc) > 8 ? 8 : $(nproc) ))
    clangd-tidy -p "$BUILD_DIR" -j "$(nproc)" --fail-on-severity warn {{ color_always }} "${headers[@]}"
    clangd-tidy -p "$BUILD_DIR" -j "$jobs_sources" --fail-on-severity warn {{ color_always }} "${sources[@]}"
    echo "clangd-tidy lint clean (${#headers[@]} headers, ${#sources[@]} sources)"

# Ensures that one or more required commands are installed
_ensure-command +command:
    #!/usr/bin/env bash
    set -euo pipefail

    read -r -a commands <<< "{{ command }}"

    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" > /dev/null 2>&1 ; then
            printf "Couldn't find required executable '%s'\n" "$cmd" >&2
            exit 1
        fi
    done

# Ensures that the CMakeCache.txt file exists
_ensure-cmake-configured:
    test -f "$BUILD_DIR"/CMakeCache.txt || just configure

# Ensures that the compile_commands.json file exists
_ensure-compile-commands-json:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command cmake
    just _ensure-cmake-configured

    if [[ ! -f "$BUILD_DIR"/compile_commands.json ]]; then
        cmake -B "$BUILD_DIR" -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    fi
