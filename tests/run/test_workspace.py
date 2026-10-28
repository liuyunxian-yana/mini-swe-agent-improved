import json
import os
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(("explicit_workspace",), [(True,), (False,)])
def test_workspace_context_binds_execution_prompt_and_trajectory(tmp_path, explicit_workspace):
    workspace = tmp_path / "case"
    workspace.mkdir()
    config = tmp_path / "deterministic.yaml"
    trajectory = tmp_path / "run.traj.json"
    config.write_text(
        yaml.safe_dump(
            {
                "agent": {"agent_class": "default", "cost_limit": 0},
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [
                        {"role": "assistant", "content": "check", "extra": {"actions": [{"command": "pwd"}]}},
                        {
                            "role": "assistant",
                            "content": "finish",
                            "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]},
                        },
                    ],
                },
            }
        )
    )
    command = [
        sys.executable,
        "-m",
        "minisweagent",
        "-c",
        "mini.yaml",
        "-c",
        "quixbugs_workspace.yaml",
        "-c",
        str(config),
        "-o",
        str(trajectory),
        "-t",
        "Check the workspace",
    ]
    if explicit_workspace:
        command.extend(["--workspace", str(workspace)])
    subprocess.run(
        command,
        cwd=tmp_path,
        env=os.environ | {"MSWEA_CONFIGURED": "true", "MSWEA_SILENT_STARTUP": "1"},
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(trajectory.read_text())
    root = workspace if explicit_workspace else tmp_path
    assert data["info"]["config"]["environment"]["cwd"] == str(root)
    assert str(root) in data["messages"][0]["content"]
    assert any(message.get("extra", {}).get("raw_output", "").strip() == str(root) for message in data["messages"])
    assert data["info"]["exit_status"] == "Submitted"
