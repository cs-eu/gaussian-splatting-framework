import config
from implementations.base import Base
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
import shutil
import os
import logging

logger = logging.getLogger(__name__)


class Viewer(Base):
    """
    Viewer class for Gaussian Splatting framework.
    This class is responsible for visualizing the results of Gaussian Splatting.
    """

    REPO_DIR = config.VIEWER_DIR
    REPO_URL = "https://github.com/nerfstudio-project/gsplat"
    COMMIT_HASH = "0b4dddf04cb687367602c01196913cde6a743d70"

    def __init__(self):
        super().__init__()
        self.repo_manager = RepoManager(self.REPO_DIR, self.REPO_URL, self.COMMIT_HASH)
        self.env_manager = CondaManager(env_name="viewer")

        # Ensure the correct viewer scripts are copied if langsplat is used
        if self.check_repo_exists():
            shutil.copyfile("implementations/lib/viewer/simple_viewer.py", f"{self.repo_manager.repo_dir}/examples/simple_viewer.py")
            shutil.copyfile("implementations/lib/viewer/gsplat_viewer.py", f"{self.repo_manager.repo_dir}/examples/gsplat_viewer.py")

    def check_env_exists(self) -> bool:
        """Check if the environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the environment."""
        self.env_manager.create_empty(python_version="3.10")
        self.env_manager.run_command("pip uninstall torch -y")
        self.env_manager.run_command(
            "pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118"
        )
        self.env_manager.run_command(
            f"pip install -r {self.REPO_DIR}/examples/requirements.txt"
        )
        self.env_manager.run_command(
            f"pip install -e {self.REPO_DIR}/."
        )
        self.env_manager.run_command(
            "pip install plyfile"
        )
        self.env_manager.run_command("pip install plyfile")
        self.env_manager.run_command("pip install open_clip_torch")

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        """Clone or create the repository."""
        self.repo_manager.clone_repo()

        # copy the modified viewer scripts
        shutil.copyfile("implementations/lib/viewer/simple_viewer.py", f"{self.repo_manager.repo_dir}/examples/simple_viewer.py")
        shutil.copyfile("implementations/lib/viewer/gsplat_viewer.py", f"{self.repo_manager.repo_dir}/examples/gsplat_viewer.py")

        src_gsplat = os.path.join(
            os.path.dirname(__file__), "lib", "viewer", "gsplat_viewer.py"
        )
        dst_gsplat = os.path.join(
            self.repo_manager.repo_dir, "examples", "gsplat_viewer.py"
        )
        shutil.copyfile(src_gsplat, dst_gsplat)

        src_exporter = os.path.join(
            os.path.dirname(__file__), "lib", "viewer", "exporter.py"
        )
        dst_exporter = os.path.join(
            self.repo_manager.repo_dir, "gsplat" , "exporter.py"
        )
        shutil.copyfile(src_exporter, dst_exporter)

        src_rendering = os.path.join(
            os.path.dirname(__file__), "lib", "viewer", "rendering.py"
        )
        dst_rendering = os.path.join(
            self.repo_manager.repo_dir, "gsplat", "rendering.py"
        )
        shutil.copyfile(src_rendering, dst_rendering)

    def run(self, path_to_ckpt: str, live_update: bool = False):
        logger.info(
            "Please start your shell with: ssh -L 8080:localhost:8080 ... to enable port forwarding for the viewer."
        )
        if path_to_ckpt:
            scene_name = os.path.basename(os.path.dirname(path_to_ckpt)).split('_')[0]
            ae_ckpt = f"ckpt/{scene_name}/best_ckpt.pth"
            command = f"python {self.REPO_DIR}/examples/simple_viewer.py --scene_grid 5 --ckpt {path_to_ckpt} --ae_ckpt {ae_ckpt}"
            if live_update:
                command += " --live_update"
            self.env_manager.run_command(command)
        else:
            command = f"python {self.REPO_DIR}/examples/simple_viewer.py --scene_grid 5"
            if live_update:
                command += " --live_update"
            self.env_manager.run_command(command, silent=False)
