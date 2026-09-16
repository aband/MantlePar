#!/usr/bin/env bash
# Place this file in the MantlePar repository root.
set -euo pipefail

# Extend this list as new modules add generated files or directories.
# Use literal names at the repository root: no wildcards or slashes.
CLEAN_TARGETS=(
    # CMake build directories (also contain CTest output).
    "build"
    "build-debug"
    "build-release"

    # Mesh output directory and the exporter's default output files.
    "output"
    "mesh.h5"
    "mesh.xdmf"

    # Executable produced by the standalone compilation command.
    "export_mesh"

    # Future examples:
    # "results"
    # "logs"
)

dry_run=false
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) dry_run=true ;;
        -h|--help)
            printf 'Usage: bash clean.sh [--dry-run]\n'
            printf 'Remove the generated paths listed in CLEAN_TARGETS.\n'
            printf 'Also remove directories named output anywhere under example/.\n'
            printf 'Also remove *.log files throughout the repository, excluding .git/.\n'
            printf 'Paths are relative to this script, regardless of your working directory.\n'
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ ! -f "$repo_root/CMakeLists.txt" || ! -d "$repo_root/src/core" ]]; then
    printf 'Place clean.sh in the MantlePar root, alongside CMakeLists.txt and src/core/.\n' >&2
    exit 1
fi

# Validate the entire list before removing anything. Restricting entries to
# top-level names also avoids traversing symlinked parent directories.
for target in "${CLEAN_TARGETS[@]}"; do
    case "$target" in
        ""|"."|".."|*/*)
            printf 'Invalid CLEAN_TARGETS entry: %s (use a top-level name).\n' "$target" >&2
            exit 1
            ;;
    esac
done

remove_path() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || return 0
    if [[ "$dry_run" == true ]]; then
        printf 'Would remove: %s\n' "$path"
    else
        printf 'Removing: %s\n' "$path"
        # No trailing slash: a symlink is removed without following its target.
        rm -rf -- "$path"
    fi
}

for target in "${CLEAN_TARGETS[@]}"; do
    remove_path "$repo_root/$target"
done

# Discover case outputs automatically as new example folders are added.
# Match the exact name output. Do not follow symlinked parent directories;
# an output symlink itself is removed without deleting its target.
example_root="$repo_root/example"
if [[ -d "$example_root" && ! -L "$example_root" ]]; then
    while IFS= read -r -d '' path; do
        remove_path "$path"
    done < <(find -P "$example_root" -mindepth 1 \
        \( -type d -o -type l \) -name output -prune -print0)
fi

# Remove log files throughout the repository, including every example case.
# Keep Git metadata and do not follow directory symlinks. Log symlinks
# themselves are removed without deleting their targets.
while IFS= read -r -d '' path; do
    remove_path "$path"
done < <(find -P "$repo_root" -name .git -prune -o \
    \( -type f -o -type l \) -name '*.log' -print0)

if [[ "$dry_run" == true ]]; then
    printf 'Preview complete; nothing was removed.\n'
else
    printf 'Cleanup complete.\n'
fi
