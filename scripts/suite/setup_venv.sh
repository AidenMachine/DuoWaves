#!/bin/bash
#
# Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International
#
# Creates the Python virtual environment used on the laptop to build the
# results spreadsheet, placing it in multicore-relation/.venv and installing
# the odfpy dependency of build_ods.py into it.
#

set -e

#
# TR-6
# Builds the virtual environment next to the results it will process and
# installs the spreadsheet library, so the .ods can be produced with
# .venv/bin/python scripts/suite/build_ods.py from any working directory.
#
main()
{
    local suite_dir relation_dir
    suite_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    relation_dir="$(cd "$suite_dir/../.." && pwd)"
    python3 -m venv "$relation_dir/.venv"
    "$relation_dir/.venv/bin/pip" install --quiet --upgrade pip odfpy
    echo "virtual environment ready: $relation_dir/.venv"
    echo "build the spreadsheet with: $relation_dir/.venv/bin/python $suite_dir/build_ods.py"
}

main
