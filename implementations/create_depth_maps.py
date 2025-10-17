import config
from implementations.base import Base
from implementations.original_gaussian_splatting import OriginalGaussianSplatting
from utils.command_runner import run_command
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
import os


class CreateDepthMaps(Base):
    """Original Gaussian Splatting implementation."""

    GS_REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting"
    GS_COMMIT_HASH = "54c035f7834b564019656c3e3fcc3646292f727d"
    DEPTH_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
    DEPTH_COMMIT_HASH = "e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1"
    NETWORK_BASE = "/mnt/projects/mlmi/gaussian_splatting"
    GS_REPO = OriginalGaussianSplatting.REPO_DIR_ORIGINAL
    DEPTH_REPO = config.DEPTH_MAPS_DIR

    def __init__(self):
        super().__init__()

        self.gs_repo_manager = RepoManager(
            repo_dir=self.GS_REPO, repo_url=self.GS_REPO_URL, commit_hash=self.GS_COMMIT_HASH
        )
        self.depth_repo_manager = RepoManager(
            repo_dir=self.DEPTH_REPO, repo_url=self.DEPTH_REPO_URL, commit_hash=self.DEPTH_COMMIT_HASH
        )
        self.env_manager = CondaManager(env_name="gaussian_splatting")

    def check_env_exists(self) -> bool:
        """Check if the conda environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the conda environment."""
        env_file = f"{self.GS_REPO}/environment.yml"
        self.env_manager.create(env_file=env_file, check=False)
        self.env_manager.run_command("pip install mkl==2024.0")
        self.env_manager.run_command(f"conda env update --file {env_file}")

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return (
            self.gs_repo_manager.check_repo_exists()
            and self.depth_repo_manager.check_repo_exists()
        )

    def create_repo(self):
        """Clone or create the repository."""
        self.gs_repo_manager.clone_repo()
        self.depth_repo_manager.clone_repo()

        # Create checkpoints directory if it doesn't exist
        checkpoints_dir = os.path.join(self.DEPTH_REPO, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Download the model checkpoint
        checkpoint_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        checkpoint_path = os.path.join(checkpoints_dir, "depth_anything_v2_vitl.pth")
        if not os.path.exists(checkpoint_path):
            run_command(f"cd {checkpoints_dir} && wget {checkpoint_url}")

        # Install requirements for Depth Anything repo
        requirements_file = os.path.join(self.DEPTH_REPO, "requirements.txt")
        run_command(f"pip install -r {requirements_file}")

    def run(self):
        """Run the Gaussian Splatting implementation."""
        dataset_paths = config.get_all_dataset_paths()

        for scenario, scenario_path in dataset_paths:
            images_to_process = "images"  # Adjust if IMAGES_TO_PROCESS is different
            depths_outdir = os.path.join(scenario_path, "depths2")

            # Run Depth Anything inference
            run_command(
                f"cd {self.DEPTH_REPO} && python run.py --encoder vitl --pred-only --grayscale "
                f"--img-path {os.path.join(scenario_path, images_to_process)} "
                f"--outdir {depths_outdir}"
            )

            # Run make_depth_scale.py
            self.env_manager.run_command(
                f"python {os.path.join(self.GS_REPO, 'utils', 'make_depth_scale.py')} "
                f"--base_dir {scenario_path} --depths_dir {depths_outdir}"
            )
