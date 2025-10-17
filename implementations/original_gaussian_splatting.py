import config
from implementations.gsbase import GSBase
from utils.conda_manager import CondaManager
from utils.pointcloud_watcher import start_pointcloud_watcher
from utils.repo_manager import RepoManager
import os
import time
from utils.command_runner import run_command
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OriginalGaussianSplatting(GSBase):
    """Original Gaussian Splatting implementation."""

    REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting"
    COMMIT_HASH = "54c035f7834b564019656c3e3fcc3646292f727d"
    NETWORK_BASE = config.GS_DIR
    REPO_DIR_ORIGINAL = f"{NETWORK_BASE}/original_gs_repo"
    REPO_DIR_FAST = f"{NETWORK_BASE}/original_gs_fast_repo"
    OUTPUT_PATH = f"{NETWORK_BASE}/output_original_gaussian_splatting"
    CURRENT_POINT_CLOUD_DIR = f"{OUTPUT_PATH}/current_point_cloud"

    def __init__(
        self,
        fast: bool = False,
        use_depth: bool = False,
        use_expcomp: bool = False,
        use_viewer: bool = False,
        resolution: int = None,
    ):
        super().__init__(use_viewer=use_viewer)
        self.fast = fast
        self.use_depth = use_depth
        self.use_expcomp = use_expcomp
        self.resolution = resolution
        if self.fast:
            self.repo_dir = self.REPO_DIR_FAST
        else:
            self.repo_dir = self.REPO_DIR_ORIGINAL

        self.repo_manager = RepoManager(repo_dir=self.repo_dir, repo_url=self.REPO_URL, commit_hash=self.COMMIT_HASH)
        self.env_manager = CondaManager(env_name="gaussian_splatting")
        self.script_path = f"{self.repo_dir}/full_eval.py"
        self.train_script_path = f"{self.repo_dir}/train.py"
        self.dataset_paths = {
            "360": config.MIPNERF360_DATASET,
            "tandt": config.TANDT_DATASET,
            "db": config.DB_DATASET,
        }

    def check_env_exists(self) -> bool:
        """Check if the conda environment exists."""
        return self.env_manager.exists()

    def create_env(self):
        """Create the conda environment."""
        if self.fast:
            submodule_dir = f"{self.repo_dir}/submodules/diff-gaussian-rasterization"
            run_command("git checkout 3dgs_accel", cwd=submodule_dir)
            run_command("rm -rf build/", cwd=submodule_dir)

            # changing line 29 of submodules/diff-gaussian-rasterization/setup.py, see https://github.com/graphdeco-inria/gaussian-splatting/issues/41#issuecomment-1752279620
            run_command(
                'sed -i \'s|extra_compile_args=.*|extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})|\' setup.py',
                cwd=submodule_dir,
            )

        env_file = f"{self.repo_dir}/environment.yml"
        self.env_manager.create(env_file=env_file, check=False, silent=True)
        self.env_manager.run_command("pip install mkl==2024.0")
        self.env_manager.run_command(f"conda env update --file {env_file}")

    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        return self.repo_manager.check_repo_exists()

    def create_repo(self):
        """Clone or create the repository."""
        self.repo_manager.clone_repo()

    def train(
        self,
        scene_path: str,
        output_path: str,
        checkpoint_iterations: list[int] = None,
        image_arg: str = None,
    ) -> str:
        """Train the model on a specific scene, with optional viewer support."""
        scene = os.path.basename(scene_path.rstrip("/"))

        logger.info(f"Starting training for scene '{scene}' at '{scene_path}'. Output will be saved to '{output_path}'.")

        common_args = "--disable_viewer --quiet --eval --test_iterations -1 "
        if self.fast:
            common_args += " --optimizer_type sparse_adam "
        if self.use_depth:
            common_args += " -d depths2/ "
        if self.use_expcomp:
            common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "
        if self.resolution:
            common_args += f" --resolution {self.resolution} "
        if image_arg:
            common_args += f" -i {image_arg} "
        if checkpoint_iterations:
            common_args += f" --checkpoint_iterations {' '.join(map(str, checkpoint_iterations))} "

        # Start the viewer if necessary
        if self.use_viewer and not self.is_viewer_running():
            save_iterations = list(range(0, 30001, 1000))
            common_args += f" --save_iterations {' '.join(map(str, save_iterations))} "
            ply_path = os.path.join(self.CURRENT_POINT_CLOUD_DIR, "point_cloud.ply")
            self.start_viewer(ply_path)

        watcher_thread = None
        stop_flag = None
        if self.use_viewer:
            point_cloud_dir = os.path.join(output_path, "point_cloud")
            stop_flag, watcher_thread = start_pointcloud_watcher(
                point_cloud_dir, self.CURRENT_POINT_CLOUD_DIR
            )

        cmd = f"cd {self.repo_dir} && python train.py -s {scene_path} -m {output_path} {common_args}"
        self.env_manager.run_command(cmd)

        if self.use_viewer and stop_flag and watcher_thread:
            stop_flag.set()
            watcher_thread.join()
        
        return output_path

    def _train_scenes(
        self, scenes, dataset_path, output_path, image_arg=None
    ):
        """Helper to train a list of scenes."""
        for scene in scenes:
            source = f"{dataset_path}/{scene}"
            self.train(
                scene_path=source,
                output_path=os.path.join(output_path, scene),
                image_arg=image_arg,
            )

    def _render_scenes(self, scenes, sources, output_path, common_args_render):
        """Helper to render a list of scenes."""
        for scene, source in zip(scenes, sources):
            cmd = f"cd {self.repo_dir} && python render.py --iteration 30000 -s {source} -m {output_path}/{scene}{common_args_render}"
            self.env_manager.run_command(cmd)

    def run(self):
        """Run the Gaussian Splatting implementation, replicating the original full eval script logic."""
        mipnerf360_outdoor_scenes = [
            "bicycle",
            "flowers",
            "garden",
            "stump",
            "treehill",
        ]
        mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
        tanks_and_temples_scenes = ["truck", "train"]
        deep_blending_scenes = ["drjohnson", "playroom"]

        all_scenes = (
            mipnerf360_outdoor_scenes
            + mipnerf360_indoor_scenes
            + tanks_and_temples_scenes
            + deep_blending_scenes
        )

        m360 = self.dataset_paths["360"]
        tandt = self.dataset_paths["tandt"]
        db = self.dataset_paths["db"]
        output_path = self.OUTPUT_PATH

        # Training
        timings = {}
        start_time = time.time()
        self._train_scenes(
            mipnerf360_outdoor_scenes,
            m360,
            output_path,
            image_arg="images_4",
        )
        self._train_scenes(
            mipnerf360_indoor_scenes,
            m360,
            output_path,
            image_arg="images_2",
        )
        timings["m360"] = (time.time() - start_time) / 60.0

        start_time = time.time()
        self._train_scenes(tanks_and_temples_scenes, tandt, output_path)
        timings["tandt"] = (time.time() - start_time) / 60.0

        start_time = time.time()
        self._train_scenes(deep_blending_scenes, db, output_path)
        timings["db"] = (time.time() - start_time) / 60.0

        timing_path = os.path.join(output_path, "timing.txt")
        with open(timing_path, "w") as file:
            for k, v in timings.items():
                file.write(f"{k}: {v} minutes \n")

        # Rendering
        all_sources = (
            [
                f"{m360}/{scene}"
                for scene in mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes
            ]
            + [f"{tandt}/{scene}" for scene in tanks_and_temples_scenes]
            + [f"{db}/{scene}" for scene in deep_blending_scenes]
        )

        common_args_render = " --quiet --eval --skip_train"
        if self.fast:
            common_args_render += " --optimizer_type sparse_adam "
        if self.use_expcomp:
            common_args_render += " --train_test_exp "

        self._render_scenes(all_scenes, all_sources, output_path, common_args_render)

        # Metrics
        scenes_string = " ".join([f'"{output_path}/{scene}"' for scene in all_scenes])
        cmd = f"cd {self.repo_dir} && python metrics.py -m {scenes_string}"
        self.env_manager.run_command(cmd)
