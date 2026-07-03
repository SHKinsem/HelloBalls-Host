from __future__ import annotations

import os
import signal
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RosLaunchConfig:
    workspace: Path = Path("ros2_ws")
    package: str = "helloballs_bringup"
    launch_file: str = "camera.launch.py"
    ros_setup: Path = Path("/opt/ros/humble/setup.bash")
    ros_log_dir: Path = Path("/tmp/ros-log")
    launch_arguments: dict[str, str] = field(default_factory=dict)


class RosLaunchProcess:
    def __init__(self, config: RosLaunchConfig) -> None:
        self.config = config
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        workspace = self.config.workspace.resolve()
        install_setup = workspace / "install" / "setup.bash"
        if not workspace.exists():
            raise FileNotFoundError(f"ROS2 workspace not found: {workspace}")
        if not install_setup.exists():
            raise FileNotFoundError(
                f"ROS2 workspace setup not found: {install_setup}. "
                "Build the workspace with colcon first."
            )
        self.config.ros_log_dir.mkdir(parents=True, exist_ok=True)

        ros2_args = [
            "exec",
            "ros2",
            "launch",
            self.config.package,
            self.config.launch_file,
            *[
                f"{name}:={value}"
                for name, value in self.config.launch_arguments.items()
                if value != ""
            ],
        ]
        command = (
            f"source {shlex.quote(str(self.config.ros_setup))} && "
            f"source {shlex.quote(str(install_setup))} && "
            + " ".join(shlex.quote(arg) for arg in ros2_args)
        )
        self.process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=workspace,
            env={**os.environ, "ROS_LOG_DIR": str(self.config.ros_log_dir)},
            start_new_session=True,
        )

    def poll(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return

        try:
            os.killpg(self.process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=3.0)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        self.process.wait(timeout=3.0)
