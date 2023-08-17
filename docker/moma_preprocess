#! /usr/bin/env python3

import os
import subprocess
import sys
from shutil import which
from pathlib import Path
from typing import List


"""These are arguments to MoMA that defined a path. We need to mount them or their parent paths in the container."""
path_args = ['-gldtp', '--gl_detection_template_path', '-i', '--input', '-o', '--output', '-log', '--logfile', '-ff', '--flatfieldpath']


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def get_bind_mount_arg(path, container_engine):
    if container_engine == "singularity":
        return f'--bind {get_directory_path(path)}'
    elif container_engine == "docker":
        return f'--mount type=bind,src={get_directory_path(path)},target={get_directory_path(path)}'
    else:
        raise ValueError(f"ERROR: Invalid containerization value: {container_engine}\n", file=sys.stderr)


def get_directory_path(target_path):
    """This function takes the path of a file or directory. If file-path is passed it returns the path to the parent
    directory. If a directory is passed, it returns the path to the directory itself."""
    if not os.path.exists(target_path):
        print(f"ERROR: Path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(target_path):
        return Path(target_path)
    elif os.path.isfile(target_path):
        return Path(os.path.dirname(target_path))


def get_mount_paths_from_args(args):
    mount_paths = []
    for ind, arg in enumerate(args):
        if arg in path_args:
            mount_paths += [get_directory_path(args[ind+1])]
    return mount_paths


def is_parent_path(parent_path, child_path):
    parent = Path(parent_path)
    child = Path(child_path)
    return child.is_relative_to(parent) and not child == parent


def get_top_level_paths(paths: List[Path]):
    top_level_paths = set(paths.copy())
    for path1 in paths:
        for path2 in paths:
            if is_parent_path(path1, path2) and path2 in top_level_paths:
                top_level_paths.remove(path2)
    return list(top_level_paths)


def build_mount_args(mount_paths: List[Path], container_engine: str):
    mount_args = []
    for path in mount_paths:
        mount_args += [get_bind_mount_arg(path, container_engine)]
    return mount_args


def get_path_from_env_var(var_name: str):
    fail_non_existing_env_var(var_name)
    return get_directory_path(os.environ.get(var_name))


def fail_non_existing_env_var(var_name: str):
    if not os.environ.get(var_name):
        print(f"ERROR: Environment variable '{var_name}' not set.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]

    if is_tool("singularity"):
        container_engine = "singularity"
    elif is_tool("docker"):
        container_engine = "docker"
    else:
        print("ERROR: No supported containerization tool found. Please install Docker or Singularity.\n", file=sys.stderr)
        sys.exit(1)

    mount_paths = get_mount_paths_from_args(args)

    # Add path from environment variables
    mount_paths += [get_path_from_env_var("HOME")]

    mount_paths = get_top_level_paths(mount_paths)

    mount_args = build_mount_args(mount_paths, container_engine)
    mount_args_string = " ".join(mount_args)

    if container_engine == "singularity":
        fail_non_existing_env_var('SINGULARITY_CONTAINER_FILE_PATH')
        singularity_container_file_path = os.environ.get("SINGULARITY_CONTAINER_FILE_PATH")
        subprocess.run(["singularity", "run",
                        *mount_args_string.split(),
                        singularity_container_file_path,
                        *args])
    elif container_engine == "docker":
        fail_non_existing_env_var('CONTAINER_TAG')
        container_tag = os.environ.get("CONTAINER_TAG")
        print(f"CONTAINER_TAG: {container_tag}")
        subprocess.run(["docker", "run", "-it", "--rm",
                        f"--user={os.getuid()}:{os.getgid()}",
                        "--env=HOME",
                        *mount_args_string.split(),
                        container_tag,
                        *args])
