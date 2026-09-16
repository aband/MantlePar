#!/usr/bin/env python3
"""Launch one or more case folders with the exact MPI launcher in their YAML."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

class UniqueLoader(yaml.SafeLoader):
    pass

def mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result

UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)

def read_config(path):
    return yaml.load(path.read_text(), Loader=UniqueLoader)

def input_path(case):
    path = Path(case).expanduser().resolve()
    return path / "input.yaml" if path.is_dir() else path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="Case folders or input.yaml paths")
    parser.add_argument("--exe", type=Path, help="Built shared mantle_driver executable")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args, petsc = parser.parse_known_args()
    if petsc and petsc[0] == "--":
        petsc = petsc[1:]
    if "-input" in petsc:
        parser.error("Select the input through the positional case argument.")
    if not args.plot_only and not args.exe:
        parser.error("--exe is required unless --plot-only is used")
    executable = args.exe.expanduser().resolve() if args.exe else None
    if executable and (not executable.is_file() or not os.access(executable, os.X_OK)):
        parser.error(f"Executable does not exist or is not executable: {executable}")
    for case in args.cases:
        path = input_path(case)
        config = read_config(path)
        if not args.plot_only:
            parallel = config["parallel"]
            launcher = Path(parallel["launcher"]).expanduser()
            if not launcher.is_absolute():
                launcher = path.parent / launcher
            if not launcher.is_file() or not os.access(launcher, os.X_OK):
                raise FileNotFoundError(f"Set parallel.launcher to PETSc's MPICH launcher: {launcher}")
            command = [str(launcher), "-n", str(parallel["ranks"]), str(executable), "-input", str(path), *petsc]
            print(shlex.join(command), flush=True)
            with (path.parent / "run.log").open("w") as log:
                log.write(shlex.join(command) + "\n")
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    status = process.wait()
                if status:
                    return status
        if not args.no_plots:
            from plot_results import plot_case
            plot_case(path, config)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"Example runner: {error}", file=sys.stderr)
        raise SystemExit(2)
