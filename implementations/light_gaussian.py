import config
from implementations.gsbase import GSBase
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
from utils.pointcloud_watcher import start_pointcloud_watcher
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LightGaussian(GSBase):
    REPO_URL = "https://github.com/VITA-Group/LightGaussian"
    COMMIT_HASH = "6676b983e77baadd909effc56a6aaadafa964dcc"
    STORAGE_BASE = config.LIGHT_GS_DIR
    REPO_DIR = f"{STORAGE_BASE}/light_gaussian_repo"
    OUTPUT_PATH = f"{STORAGE_BASE}/output"
    CURRENT_POINT_CLOUD_DIR = f"{OUTPUT_PATH}/current_point_cloud"

    def __init__(self, use_viewer: bool = False):
        super().__init__(use_viewer=use_viewer)

        self.repo_manager = RepoManager(repo_dir=self.REPO_DIR, repo_url=self.REPO_URL, commit_hash=self.COMMIT_HASH)
        self.env_manager = CondaManager(env_name="lightgaussian")
        self.train_script_path = f"{self.REPO_DIR}/train_densify_prune.py"
        self.render_script_path = f"{self.REPO_DIR}/render.py"
        self.metrics_script_path = f"{self.REPO_DIR}/metrics.py"

    def check_env_exists(self) -> bool:
        """Check if the environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the environment."""
        env_file = f"{self.REPO_DIR}/environment.yml"
        self.env_manager.create(env_file=env_file, check=False)
        self.env_manager.run_command("pip install mkl==2024.0")
        self.env_manager.run_command('pip install "numpy<2"')
        self.env_manager.run_command(f"conda env update --file {env_file}")

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        """Clone or create the repository."""
        self.repo_manager.clone_repo()

    def train(self, scene_path: str, output_path: str, checkpoint_iterations: list[int] = None) -> str:
        """Train the Gaussian Splatting model."""
        special_args = ""
        if self.use_viewer and not self.is_viewer_running():
            save_iterations = list(range(0, 30001, 1000))
            special_args += f"--save_iterations {' '.join(map(str, save_iterations))} "
            ply_path = os.path.join(self.CURRENT_POINT_CLOUD_DIR, "point_cloud.ply")
            self.start_viewer(ply_path)

        if checkpoint_iterations:
            special_args += f"--checkpoint_iterations {' '.join(map(str, checkpoint_iterations))} "

        # Start watcher before training
        if self.use_viewer:
            point_cloud_dir = os.path.join(output_path, "point_cloud")
            stop_flag, watcher_thread = start_pointcloud_watcher(point_cloud_dir, self.CURRENT_POINT_CLOUD_DIR)

        command = f"python {self.train_script_path} -s {scene_path} --model_path {output_path} {special_args}"
        self.env_manager.run_command(command, check=True)

        # Stop watcher after training
        if self.use_viewer:
            stop_flag.set()
            watcher_thread.join()

        return output_path

    def run(self, *args, **kwargs):
        """Run the Gaussian Splatting implementation."""
        dataset_paths = config.get_all_dataset_paths()

        for scenario, scenario_path in dataset_paths:
            logger.info(f"Processing dataset: {scenario_path}")

            # Step 1: Train    
            self.train(scenario_path, f"{self.OUTPUT_PATH}/{scenario}")      

            # Step 2: Render with evaluation
            command = f"python {self.render_script_path} --source_path {scenario_path} --model_path {self.OUTPUT_PATH}/{scenario} --eval"
            self.env_manager.run_command(command, check=True)

            # Step 3: Organize files for metric computation
            render_dir = f"{self.OUTPUT_PATH}/{scenario}/test/ours_30000"
            self.env_manager.run_command(
                f'mkdir -p "{render_dir}/renders" "{render_dir}/gt"', check=True
            )
            self.env_manager.run_command(
                f'mv "{render_dir}"/*.png "{render_dir}/renders/" 2>/dev/null',
                check=False,
            )
            self.env_manager.run_command(
                f'cp "{scenario_path}/images/"*.png "{render_dir}/gt/" 2>/dev/null',
                check=False,
            )

            # Step 4: Compute metrics
            command = f"python {self.metrics_script_path} --model_paths {self.OUTPUT_PATH}/{scenario}"
            self.env_manager.run_command(command, check=True)
