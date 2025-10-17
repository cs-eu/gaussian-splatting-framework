import config
from implementations.gsbase import Base
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
from utils.pointcloud_watcher import start_pointcloud_watcher
import os
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class EndoGaussian(Base):
    REPO_URL = "https://github.com/yifliu3/EndoGaussian.git"
    STORAGE_BASE = config.ENDO_GAUSSIAN_DIR
    REPO_DIR = f"{STORAGE_BASE}/endo_gaussian_repo"
    OUTPUT_PATH = f"{STORAGE_BASE}/output"
    CURRENT_POINT_CLOUD_DIR = f"{OUTPUT_PATH}/current_point_cloud"

    def __init__(self, pulling: bool = True, cutting: bool = True):
        super().__init__()

        self.repo_manager = RepoManager(repo_dir=self.REPO_DIR, repo_url=self.REPO_URL)
        self.env_manager = CondaManager(env_name="endogaussian")
        self.train_script_path = f"{self.REPO_DIR}/train.py"
        self.render_script_path = f"{self.REPO_DIR}/render.py"
        self.pulling = pulling
        self.cutting = cutting

    def check_env_exists(self) -> bool:
        """Check if the environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the environment."""
        requirements_file = f"{self.REPO_DIR}/requirements.txt"
        self.env_manager.create_empty(python_version="3.7")
        self.env_manager.run_command(f"pip install -r {requirements_file}", silent=False)
        self.env_manager.run_command("pip install mkl==2024.0")
        self.env_manager.run_command(f"cd {self.REPO_DIR} && git submodule update --init --recursive", silent=False)
        self.env_manager.run_command(f"cd {self.REPO_DIR}/submodules && rm -rf depth-diff-gaussian-rasterization", silent=False)
        self.env_manager.run_command(f"cd {self.REPO_DIR}/submodules && git clone  --recursive https://github.com/ingra14m/depth-diff-gaussian-rasterization.git", silent=False)
        self.env_manager.run_command(f"cd ..", silent=False)
        self.env_manager.run_command(f"cd {self.REPO_DIR} && pip install -e submodules/depth-diff-gaussian-rasterization", silent=False)
        self.env_manager.run_command(f"cd {self.REPO_DIR} && pip install -e submodules/simple-knn", silent=False)
        self.env_manager.run_command("pip install open3d", silent=False)
        self.env_manager.run_command("pip install imageio", silent=False)

        src = os.path.join(
            os.path.dirname(__file__), "lib", "endogaussian", "render.py"
        )
        dst = os.path.join(
            self.repo_manager.repo_dir, "render.py"
        )
        shutil.copyfile(src, dst)

        src = os.path.join(
            os.path.dirname(__file__), "lib", "endogaussian", "__init__.py"
        )
        dst = os.path.join(
            self.repo_manager.repo_dir, "gaussian_renderer", "__init__.py"
        )
        shutil.copyfile(src, dst)

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        """Clone or create the repository."""
        self.repo_manager.clone_repo()


    def run(self, *args, **kwargs):
        """Run the Gaussian Splatting implementation."""
        train_cutting_and_pulling = (self.pulling == False) & (self.cutting == False)

        if self.pulling or train_cutting_and_pulling:
            print("Training on pulling soft tissues...")
            self.env_manager.run_command(
                f"python {self.train_script_path} -s {config.ENDOGAUSSIAN_DATASET}/pulling_soft_tissues/ --expname {self.REPO_DIR}/endonerf/pulling --configs {self.REPO_DIR}/arguments/endonerf/pulling.py",
                silent=False,
            )
            print("Rendering on pulling soft tissues...")
            self.env_manager.run_command(
                f"python {self.render_script_path} --model_path output/endonerf/pulling --skip_train --skip_test --configs {self.REPO_DIR}/arguments/endonerf/pulling.py",
                silent=False,
            )
        if self.cutting or train_cutting_and_pulling:
            print("Training on cutting tissues...")
            self.env_manager.run_command(
                f"python {self.train_script_path} -s {config.ENDOGAUSSIAN_DATASET}/cutting_tissues_twice/ --expname {self.REPO_DIR}/endonerf/cutting --configs {self.REPO_DIR}/arguments/endonerf/cutting.py",
                silent=False,
            )
            print("Rendering on cutting tissues...")
            self.env_manager.run_command(
                f"python {self.render_script_path} --model_path output/endonerf/cutting --skip_train --skip_test --configs {self.REPO_DIR}/arguments/endonerf/cutting.py",
                silent=False,
            )
