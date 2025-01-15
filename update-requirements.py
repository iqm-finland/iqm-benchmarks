import subprocess
from packaging.requirements import Requirement
import tomllib

if __name__ == "__main__":
    print("Parsing pyproject.toml for all direct project dependencies and optional dependencies.")
    with open("pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    dependencies = pyproject["project"]["dependencies"]
    dependencies += pyproject["project"]["optional-dependencies"]["cicd"]
    dependencies += pyproject["project"]["optional-dependencies"]["examples"]
    dependencies += pyproject["project"]["optional-dependencies"]["mgst"]
    dependencies += pyproject["project"]["optional-dependencies"]["test"]
    dependencies += pyproject["project"]["optional-dependencies"]["docs"]

    # NOTE: see comment about build deps in pyproject.toml.

    update_lines = [f"--upgrade-package={Requirement(dep).name}" for dep in dependencies]
    cmd = [
        "uv",
        "pip",
        "compile",
        "--generate-hashes",
        "--no-annotate",
        "--no-emit-index-url",
        "--no-emit-find-links",
        "--no-header",
        "--no-cache",
        "--python-platform=x86_64-manylinux_2_31",
        "--all-extras",
        "pyproject.toml",
        "--output-file=requirements.txt",
    ] + update_lines
    print(f"Executing `{subprocess.list2cmdline(cmd)}`")
    subprocess.run(cmd, check=True)
