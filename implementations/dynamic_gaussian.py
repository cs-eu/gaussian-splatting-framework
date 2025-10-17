import config
from implementations.base import Base
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
import logging
from pathlib import Path
import os
import shutil

logger = logging.getLogger(__name__)


class DynamicGaussian(Base):
    REPO_URL = "https://github.com/JonathonLuiten/Dynamic3DGaussians"
    COMMIT_HASH = "7dbbd4dec404308524ff402756bdb8143a2589b0"
    STORAGE_BASE = config.DYNAMIC_GS_DIR
    REPO_DIR = f"{STORAGE_BASE}/dynamic_gaussians_repo"
    OUTPUT_PATH = f"{REPO_DIR}/output"  # matches repo default

    # The external CUDA rasterizer repo URL needed for installation
    CUDA_RASTERIZER_REPO_URL = (
        "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git"
    )
    CUDA_RASTERIZER_DIR = f"{STORAGE_BASE}/diff-gaussian-rasterization-w-depth"

    def __init__(
        self, 
        input_path: str = None,
        output_path: str = None
    ):
        super().__init__()

        self.input_path = input_path or config.DYNAMIC_GAUSSIAN_DATASET
        self.output_path = output_path or self.OUTPUT_PATH

        self.repo_manager = RepoManager(repo_dir=self.REPO_DIR, repo_url=self.REPO_URL)
        # self.repo_manager = RepoManager(repo_dir=self.REPO_DIR, repo_url=self.REPO_URL, commit_hash=self.COMMIT_HASH)
        self.env_manager = CondaManager(env_name="dynamic_gaussians")
        self.train_script_path = f"{self.REPO_DIR}/train.py"
        self.visualize_script_path = f"{self.REPO_DIR}/visualize.py"

    def check_env_exists(self) -> bool:
        return self.env_manager.exists()

    def create_env(self):
        env_file = f"{self.REPO_DIR}/environment.yml"
        self.env_manager.create(env_file=env_file, check=False)
        # Post-install: install external rasterizer
        self._install_external_rasterizer()

    def _install_external_rasterizer(self):
        import os

        if not os.path.exists(self.CUDA_RASTERIZER_DIR):
            self.env_manager.run_command(
                f"git clone {self.CUDA_RASTERIZER_REPO_URL} {self.CUDA_RASTERIZER_DIR}",
                check=True,
            )
            self.env_manager.run_command(
                f"cd {self.CUDA_RASTERIZER_DIR} && python setup.py install && pip install .",
                check=True,
            )

    def check_repo_exists(self) -> bool:
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        self.repo_manager.clone_repo()

        src = os.path.join(
            os.path.dirname(__file__), "lib", "dynamic_gaussian", "train.py"
        )
        dst = os.path.join(
            self.repo_manager.repo_dir, "train.py"
        )
        shutil.copyfile(src, dst)

        src = os.path.join(
            os.path.dirname(__file__), "lib", "dynamic_gaussian", "helpers.py"
        )
        dst = os.path.join(
            self.repo_manager.repo_dir, "helpers.py"
        )
        shutil.copyfile(src, dst)

    def run(self, *args, **kwargs):
        meta_file = os.path.join(self.input_path, "train_meta.json")

        if os.path.isfile(meta_file):
            logger.info(f"Processing dataset: {self.input_path}")
            command = f"python {self.train_script_path} --input {self.input_path} --output {self.output_path}"
            self.env_manager.run_command(command, silent=False)
        else:
            scenarios = [
                os.path.join(self.input_path, d)
                for d in os.listdir(self.input_path)
                if os.path.isdir(os.path.join(self.input_path, d))
            ]
            logger.info(f"Scenarios: {', '.join(os.path.basename(s) for s in scenarios)}")
            for scenario in scenarios:
                logger.info(f"Processing dataset: {scenario}")
                command = f"python {self.train_script_path} --input {scenario} --output {self.output_path}"
                self.env_manager.run_command(command, silent=False)


