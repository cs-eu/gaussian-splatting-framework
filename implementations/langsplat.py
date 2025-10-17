import os
import config
from implementations.base import Base
from implementations.light_gaussian import LightGaussian
from implementations.original_gaussian_splatting import OriginalGaussianSplatting
from utils.conda_manager import CondaManager
from utils.repo_manager import RepoManager
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LangSplat(Base):
    """LangSplat: Semantical conditioning of 3DGS."""

    REPO_URL = "https://github.com/minghanqin/LangSplat"
    COMMIT_HASH = "eda172836c6765d477717eb1595795a739565fbb"
    REPO_DIR = config.LANGSPLAT_DIR
    ORIG_GS_CKPT_PATH = f"{REPO_DIR}/3dgs_output"
    OUTPUT_PATH = f"{REPO_DIR}/output"

    def __init__(
        self,
        run_all: bool = False,
        preprocess: bool = False,
        train_autoencoder: bool = False,
        train_3dgs: bool = False,
        gs_impl: str = "original",
        train_semantic_features: bool = False,
        resolution: int = None,
        scenario: str = None,
        render: bool = False,
    ):
        super().__init__()
        self.repo_dir = self.REPO_DIR
        self.run_all = run_all
        self.preprocess = preprocess
        self.train_autoencoder = train_autoencoder
        self.train_3dgs = train_3dgs
        self.gs_impl = gs_impl
        self.train_semantic_features = train_semantic_features
        self.resolution = resolution
        self.scenario = scenario
        self.render = render

        if self.gs_impl == "original":
            self.gs_impl = OriginalGaussianSplatting(
                fast=True,
                resolution=self.resolution,
            )
        elif self.gs_impl == "LightGaussian":
            self.gs_impl = LightGaussian()

        self.repo_manager = RepoManager(repo_dir=self.repo_dir, repo_url=self.REPO_URL, commit_hash=self.COMMIT_HASH)
        self.env_manager = CondaManager(env_name="langsplat")

        self.preprocess_script_path = f"{self.repo_dir}/preprocess.py"
        self.ae_train_script_path = f"{self.repo_dir}/autoencoder/train.py"
        self.ae_test_script_path = f"{self.repo_dir}/autoencoder/test.py"
        self.train_script_path = f"{self.repo_dir}/train.py"
        self.render_script_path = f"{self.repo_dir}/render.py"


    def check_env_exists(self) -> bool:
        """Check if the conda environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the conda environment."""
        src_env_file = "implementations/lib/langsplat/langsplat_env.yml"
        dst_env_file = f"{self.repo_dir}/lang_splat_env.yml"

        # Change the setup file of the diff-gaussian-rasterization submodule
        src = "implementations/lib/langsplat/3dgs-accel-diff-gaussian-rasterization-setup.py"
        dst = f"{self.repo_dir}/submodules/langsplat-rasterization/setup.py"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        shutil.copyfile(src_env_file, dst_env_file)

        self.env_manager.create(env_file=dst_env_file, check=False)
        self.env_manager.run_command("pip install mkl==2024.0")
        self.env_manager.run_command('pip install "numpy<2"')
        self.env_manager.run_command(f"conda env update --file {dst_env_file}")

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        """Clone or create the repository."""
        self.repo_manager.clone_repo()

    def run(self):
        """Run the LangSplat implementation."""
        dataset_paths = config.get_all_dataset_paths(scenario=self.scenario)

        for scenario, scenario_path in dataset_paths:
            logger.info(f"Processing dataset: {scenario_path}")
            logger.info(f"Resolution: {self.resolution}")

            # Step 1: Preprocess (SAM + CLIP)
            if self.run_all or self.preprocess:
                logger.info("=" * 50)
                logger.info(" " * 15 + "PREPROCESSING (SAM + CLIP)")
                logger.info("=" * 50 + "\n")
                ckpt_dir = os.path.join(self.REPO_DIR, "ckpts")
                ckpt_filename = "sam_vit_h_4b8939.pth"
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                download_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                if not os.path.exists(ckpt_path):
                    logger.info(f"Downloading SAM checkpoint into {ckpt_path}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    command = f"curl -L {download_url} -o {ckpt_path}"
                    self.env_manager.run_command(command, check=True)
                else:
                    logger.info(f"Checkpoint already exists at {ckpt_path}")
                if isinstance(self.resolution, int):
                    command = f"python {self.preprocess_script_path} --dataset_path {scenario_path} --sam_ckpt_path {ckpt_path} --resolution {self.resolution}"
                else:
                    command = f"python {self.preprocess_script_path} --dataset_path {scenario_path} --sam_ckpt_path {ckpt_path}"

                self.env_manager.run_command(command, check=True)

            # Step 2: Train Autoencoder
            if self.run_all or self.train_autoencoder:
                logger.info("=" * 50)
                logger.info(" " * 15 + "TRAINING AUTOENCODER")
                logger.info("=" * 50 + "\n")
                command = f"python {self.ae_train_script_path} --dataset_path {scenario_path} --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name {scenario}"
                self.env_manager.run_command(command, check=True)
                command = f"python {self.ae_test_script_path} --dataset_path {scenario_path} --dataset_name {scenario}"
                self.env_manager.run_command(command, check=True)

            # Step 3: Train original 3DGS
            if self.run_all or self.train_3dgs:
                logger.info("=" * 50)
                logger.info(" " * 15 + "TRAINING 3DGS")
                logger.info("=" * 50 + "\n")
                self.gs_impl.setup()
                self.gs_impl.train(
                    scene_path=scenario_path,
                    output_path=f"{self.ORIG_GS_CKPT_PATH}/{scenario}",
                    checkpoint_iterations=[30000],
                )

            # Step 4: Train LangSplat
            if self.run_all or self.train_semantic_features:
                logger.info("=" * 50)
                logger.info(" " * 15 + "TRAINING LANGSPLAT")
                logger.info("=" * 50 + "\n")
                command = f"python {self.train_script_path} -s {scenario_path} -m {self.OUTPUT_PATH}/{scenario} --start_checkpoint {self.ORIG_GS_CKPT_PATH}/{scenario}/chkpnt30000.pth --feature_level 1"
                if self.resolution:
                    command += f" --resolution {self.resolution}"
                self.env_manager.run_command(command, check=True)

            # Step 5: Render the LangSplat
            if self.run_all or self.render:
                logger.info("=" * 50)
                logger.info(" " * 15 + "RENDERING LANGSPLAT")
                logger.info("=" * 50 + "\n")
                feature_level = 1  # You may want to make this configurable
                command = f"python {self.render_script_path} -m {self.OUTPUT_PATH}/{scenario}_{feature_level}"
                if self.resolution:
                    command += f" --resolution {self.resolution}"
                self.env_manager.run_command(command, check=True)

                # Render with features
                command += f" --include_feature"
                self.env_manager.run_command(command, check=True)
