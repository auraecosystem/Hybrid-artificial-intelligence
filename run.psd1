# Python
winget install --id=Python.Python.3.12 -e
# Taskfile.dev
winget install --id=Task.Task -e
# uv
winget install --id=astral-sh.uv  -e
# Node.js
winget install --id=OpenJS.NodeJS -e
# Pulumi
winget install pulumi
winget upgrade pulumi
# Windows Developer Tools
winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
# Visual Code Redistributable (required by DuckDB)
winget install -e --id Microsoft.VCRedist.2015+.x86

# For Windows 10/11, toggle Developer Mode to "On" under System > For developer to enable symbolic link.
# This repository uses symlinks. Set the following.
git config --global core.symlink true
# Alternatively, run the same command without --global in the repo root to scope it to this repository only.
uv sync
uv run pytest tests                # Against an existing Pulumi stack
uv run pytest tests --pulumi_up    # Run `pulumi up` first, then test
