import viser
import sys
import os
import time
import threading
from pathlib import Path
from typing import Literal, Tuple, Callable, Optional, List, Dict, Any
from nerfview import Viewer, RenderTabState
import torch

framework_dir = os.environ["FRAMEWORK_BASE"]
sys.path.insert(0, framework_dir)
from implementations.original_gaussian_splatting import OriginalGaussianSplatting
from implementations.light_gaussian import LightGaussian
from implementations.langsplat import LangSplat
from implementations.dynamic_gaussian import DynamicGaussian

CURRENT_PATH = None
ORIGINAL_PATH = os.environ["GS_DIR"] + "/output_original_gaussian_splatting/current_point_cloud/point_cloud.ply"
LIGHT_PATH = os.environ["LIGHT_GS_DIR"] + "/output/current_point_cloud/point_cloud.ply"
LANG_PATH = os.environ["LANGSPLAT_DIR"] + "/output/current_point_cloud/point_cloud.ply"
DYN_PATH = os.environ["DYNAMIC_GS_DIR"] + "/output/current_point_cloud/point_cloud.ply"


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha", "number of gaussians", "average opacity", "average scale"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    heatmap_min: float = 0.0
    heatmap_max: float = 1.0
    max_scale: float = 1.0
    compare: bool = False
    means_threshold: float = 1.0
    quats_threshold: float = 1.0
    scales_threshold: float = 1.0
    opacities_threshold: float = 1.0
    colors_threshold: float = 1.0

    original_fast: bool = False
    original_use_depth: bool = False
    original_use_expcomp: bool = False
    langsplat_text: str = ""

    fps: int = 30  
    video_stopped: bool = False


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        load_callback: Optional[Callable[[str, Optional[RenderTabState]], None]] = None,
        #load_callback: Optional[Callable[[str, Optional[RenderTabState], Optional[int], Optional[Dict[str, torch.Tensor]]], None]] = None,
        load_video_callback: Optional[Callable[[str], Tuple[Dict[str, torch.Tensor]]]] = None,
        live_update_path: str = None,
        #animation_cache_callback: Optional[Callable[[str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        export_callback: Optional[Callable[[str], None]] = None,
        model_state_callback: Optional[Callable[[RenderTabState], None]] = None,
    ):
        self.load_callback = load_callback
        global CURRENT_PATH
        CURRENT_PATH = live_update_path
        self.live_update_path = live_update_path
        #self.animation_cache_callback = animation_cache_callback
        self.load_video_callback = load_video_callback
        self.export_callback = export_callback
        self.model_state_callback = model_state_callback
        
        self._live_update_thread = None
        self._live_update_stop_event = threading.Event()
        self._live_update_last_mtime = None

        if self.live_update_path:
            self._live_update_last_mtime = None
            self._live_update_stop_event.clear()
            self._live_update_thread = threading.Thread(
                target=self._live_update_poll_loop,
                args=(1, self.rerender),
                daemon=True
            )
            self._live_update_thread.start()
        self._video_cache = None
        self._video_pause_event = threading.Event()
        self._timestep = 0
        self._num_timesteps = 10000

        self._trainer_execution_lock = threading.Lock()
        self._trainer_buttons: List[viser.GuiHandle[viser.Button]] = []

        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _live_update_poll_loop(self, poll_interval: int, rerender_cb):
        while not self._live_update_stop_event.is_set():
            if CURRENT_PATH and self.load_callback:
                current_mtime = None
                try:
                    if os.path.exists(CURRENT_PATH):
                        current_mtime = os.path.getmtime(CURRENT_PATH)
                except Exception as e:
                    print(f"[Live Update] Error checking file: {e}")
                    pass
                if current_mtime is not None and current_mtime != self._live_update_last_mtime:
                    print(f"[Live Update] Detected change in {CURRENT_PATH}. Reloading...")
                    self.load_callback(CURRENT_PATH, self.render_tab_state)
                    rerender_cb(None)
                    self._live_update_last_mtime = current_mtime
            self._live_update_stop_event.wait(timeout=poll_interval)

    def _video_loop(self, rerender_cb):
        while True:
            interval = 1.0 / self.render_tab_state.fps if self.render_tab_state.fps > 0 else 1.0 / 30.0
            self._video_pause_event.wait()
            if CURRENT_PATH and self.load_callback:
                self.load_callback(CURRENT_PATH, self.render_tab_state, timestep=self._timestep, video_cache=self._video_cache)
                rerender_cb(None)
                self._timestep = (self._timestep + 1) % (self._num_timesteps + 1)
                self.frame_slider.value = self._timestep
            time.sleep(interval)

    def _run_trainer_thread(self, trainer_instance, output_path: str):
        global CURRENT_PATH
        if not self._trainer_execution_lock.acquire(blocking=False):
            print("Another trainer is already running. Please wait for it to finish.")
            return
        try:
            print(f"Starting trainer: {trainer_instance.__class__.__name__}.")
            for button in self._trainer_buttons:
                button.disabled = True
            trainer_instance.execute()
            CURRENT_PATH = output_path
            print(f"Trainer finished. Set CURRENT_PATH to: {output_path}")
        except Exception as e:
            print(f"An error occurred during trainer execution: {e}")
        finally:
            for button in self._trainer_buttons:
                button.disabled = False
            self._trainer_execution_lock.release()

    def _populate_rendering_tab(self):
        server = self.server
        install_ori_button: Optional[viser.GuiHandle[viser.Button]] = None
        install_light_button: Optional[viser.GuiHandle[viser.Button]] = None
        install_lang_button: Optional[viser.GuiHandle[viser.Button]] = None
        install_dyn_button: Optional[viser.GuiHandle[viser.Button]] = None
        self._trainer_buttons.clear()

        with self.server.gui.add_folder("Data I/O"):
            with self.server.gui.add_folder("Load Container"):
                load_path_textbox = self.server.gui.add_text(
                    "Load Path",
                    initial_value="",
                    hint="Path to a .ply or .pt file to load."
                )
                load_button = self.server.gui.add_button(
                    "Load Data",
                    hint="Load a new Gaussian model from the specified path."
                )

                @load_button.on_click
                def _(_) -> None:
                    global CURRENT_PATH
                    path_to_load = load_path_textbox.value
                    if not path_to_load:
                        print("Load path is empty.")
                        return
                    CURRENT_PATH = path_to_load

                    if ".npz" in path_to_load:
                        print("Starting a video loop for .npz file.")
                        self._timestep = 0
                        self._video_pause_event.set()
                        threading.Thread(
                            target=self._video_loop,
                            args=(self.rerender,),
                            daemon=True
                        ).start()
                    else:
                        self.load_callback(path_to_load, self.render_tab_state)
                    self.rerender(_)
            
            with self.server.gui.add_folder("Export Container"):
                export_path_textbox = self.server.gui.add_text(
                    "Export Path",
                    initial_value="",
                    hint="Path to save the .ply or .pt file."
                )
                export_button = self.server.gui.add_button(
                    "Export",
                    hint="Export the current Gaussian model to the specified path."
                )

                @export_button.on_click
                def _(_) -> None:
                    path_to_export = export_path_textbox.value
                    if not path_to_export:
                        print("Export path is empty.")
                        return
                    self.export_callback(path_to_export)

        with self.server.gui.add_folder("Training"):
            with self.server.gui.add_folder("Live Update"):
                live_update = server.gui.add_checkbox(
                    "Update Live",
                    initial_value=False if self.live_update_path is None else True,
                    disabled=False,
                    hint="Updates the viewer whenever the model is changed.",
                )

                @live_update.on_update
                def _(_) -> None:
                    if self._live_update_thread is None:
                        self._live_update_last_mtime = None
                        self._live_update_stop_event.clear()
                        self._live_update_thread = threading.Thread(
                            target=self._live_update_poll_loop,
                            args=(1, self.rerender),
                            daemon=True
                        )
                        self._live_update_thread.start()
                    else:
                        if self._live_update_thread is not None:
                            self._live_update_stop_event.set()
                            self._live_update_thread.join()
                            self._live_update_thread = None

            with self.server.gui.add_folder("Original_GS"):
                ori_fast_box = server.gui.add_checkbox(
                    "Fast",
                    initial_value=False,
                    disabled=False,
                    hint="Enable fast implementation.",
                )

                @ori_fast_box.on_update
                def _(_) -> None:
                    self.render_tab_state.original_fast = (
                        ori_fast_box.value
                    )

                ori_depth_box = server.gui.add_checkbox(
                    "Depth",
                    initial_value=False,
                    disabled=False,
                    hint="Enable depth.",
                )

                @ori_depth_box.on_update
                def _(_) -> None:
                    self.render_tab_state.original_use_depth = (
                        ori_depth_box.value
                    )

                ori_exp_box = server.gui.add_checkbox(
                    "Expcomp",
                    initial_value=False,
                    disabled=False,
                    hint="Enable expcomp.",
                )

                @ori_exp_box.on_update
                def _(_) -> None:
                    self.render_tab_state.original_use_expcomp = (
                        ori_exp_box.value
                    )

                install_ori_button = self.server.gui.add_button(
                    "Install & Run",
                    hint="Install and run the original GS implementation."
                )
                self._trainer_buttons.append(install_ori_button)

                @install_ori_button.on_click
                def _(_) -> None:
                    print("Initializing Original GS Trainer...")
                    original_trainer = OriginalGaussianSplatting(
                        fast=self.render_tab_state.original_fast,
                        use_depth=self.render_tab_state.original_use_depth,
                        use_expcomp=self.render_tab_state.original_use_expcomp,
                        use_viewer=True
                    )
                    threading.Thread(
                        target=self._run_trainer_thread,
                        args=(original_trainer, ORIGINAL_PATH),
                        daemon=True
                    ).start()

            with self.server.gui.add_folder("LightGaussian"):
                install_light_button = self.server.gui.add_button(
                    "Install & Run",
                    hint="Install and run the LightGaussian implementation."
                )
                self._trainer_buttons.append(install_light_button)

                @install_light_button.on_click
                def _(_) -> None:
                    print("Initializing LightGaussian Trainer...")
                    light_trainer = LightGaussian(use_viewer=True)
                    threading.Thread(
                        target=self._run_trainer_thread,
                        args=(light_trainer, LIGHT_PATH),
                        daemon=True
                    ).start()

            with self.server.gui.add_folder("LangSplat"):
                install_lang_button = self.server.gui.add_button(
                    "Install & Run",
                    hint="Install and run the LangSplat implementation."
                )
                self._trainer_buttons.append(install_lang_button)

                @install_lang_button.on_click
                def _(_) -> None:
                    print("Initializing LangSplat Trainer...")
                    lang_trainer = LangSplat()
                    threading.Thread(
                        target=self._run_trainer_thread,
                        args=(lang_trainer, LANG_PATH),
                        daemon=True
                    ).start()

            with self.server.gui.add_folder("DynamicGaussian"):
                install_dyn_button = self.server.gui.add_button(
                    "Install & Run",
                    hint="Install and run the DynamicGaussian implementation."
                )
                self._trainer_buttons.append(install_dyn_button)

                @install_dyn_button.on_click
                def _(_) -> None:
                    print("Initializing DynamicGaussian Trainer...")
                    dyn_trainer = DynamicGaussian()
                    threading.Thread(
                        target=self._run_trainer_thread,
                        args=(dyn_trainer, DYN_PATH),
                        daemon=True
                    ).start()
                
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                langsplat_text_input = server.gui.add_text(
                    "LangSplat Prompt",
                    initial_value="",
                    hint="Enter prompt for LangSplat here."
                )

                @langsplat_text_input.on_update
                def _(_) -> None:
                    self.render_tab_state.langsplat_text = langsplat_text_input.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha", "number of gaussians", "average opacity", "average scale", "language similarity", "3D language similarity", "3D language features"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.visible = True
                        inverse_checkbox.visible = True
                        heatmap_range_vec2.visible = False
                    elif "average" in render_mode_dropdown.value or "number of gaussians" in render_mode_dropdown.value:
                        heatmap_range_vec2.visible = True
                        normalize_nearfar_checkbox.visible = True
                        inverse_checkbox.visible = True
                    else:
                        normalize_nearfar_checkbox.visible = False
                        inverse_checkbox.visible = False
                        heatmap_range_vec2.visible = False
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                heatmap_range_vec2 = server.gui.add_vector2(
                    "Heatmap Range",
                    initial_value=(
                        self.render_tab_state.heatmap_min,
                        self.render_tab_state.heatmap_max,
                    ),
                    min=(0, 0),
                    max=(10.0, 10.0),
                    step=0.001,
                    hint="Min and max values for heatmap visualization.",
                    visible=False,
                )

                @heatmap_range_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.heatmap_min = heatmap_range_vec2.value[0]
                    self.render_tab_state.heatmap_max = heatmap_range_vec2.value[1]
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    visible=False,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    visible=False,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                max_scale_slider = server.gui.add_slider(
                    "Max Scale (Log)",
                    min=-6.0,
                    max=0.0,
                    step=1e-3,
                    initial_value=0.0,
                    hint="Logarithmically control the maximum Gaussian scale.",
                )

                @max_scale_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.max_scale = 10**max_scale_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                compare_checkbox = server.gui.add_checkbox(
                    "Compare",
                    initial_value=False,
                    disabled=False,
                    hint="Compare Models",
                )

                @compare_checkbox.on_update
                def _(_) -> None:
                    if compare_checkbox.value:
                        means_compare_slider.visible = True
                        quats_compare_slider.visible = True
                        scales_compare_slider.visible = True
                        opacities_compare_slider.visible = True
                        colors_compare_slider.visible = True
                    else:
                        means_compare_slider.visible = False
                        quats_compare_slider.visible = False
                        scales_compare_slider.visible = False
                        opacities_compare_slider.visible = False
                        colors_compare_slider.visible = False
                    self.render_tab_state.compare = compare_checkbox.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                means_compare_slider = server.gui.add_slider(
                    "Means Threshold",
                    min=0,
                    max=1,
                    step=1e-3,
                    initial_value=1,
                    visible=False,
                    hint="Control means threshold for model comparison.",
                )

                @means_compare_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.means_threshold = means_compare_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                quats_compare_slider = server.gui.add_slider(
                    "Quaternions Threshold",
                    min=0,
                    max=1,
                    step=1e-3,
                    initial_value=1,
                    visible=False,
                    hint="Control quaternions threshold for model comparison.",
                )

                @quats_compare_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.quats_threshold = quats_compare_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                scales_compare_slider = server.gui.add_slider(
                    "Scales Threshold",
                    min=0,
                    max=1,
                    step=1e-3,
                    initial_value=1,
                    visible=False,
                    hint="Control scales threshold for model comparison.",
                )

                @scales_compare_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.scales_threshold = scales_compare_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                opacities_compare_slider = server.gui.add_slider(
                    "Opacities Threshold",
                    min=0,
                    max=1,
                    step=1e-3,
                    initial_value=1,
                    visible=False,
                    hint="Control opacities threshold for model comparison.",
                )

                @opacities_compare_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.opacities_threshold = opacities_compare_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)

                colors_compare_slider = server.gui.add_slider(
                    "Colors Threshold",
                    min=0,
                    max=1,
                    step=1e-3,
                    initial_value=1,
                    visible=False,
                    hint="Control colors threshold for model comparison.",
                )

                @colors_compare_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.colors_threshold = colors_compare_slider.value
                    self.model_state_callback(self.render_tab_state)
                    self.rerender(_)
                    
                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)
            
            with self.server.gui.add_folder("Video"):
                markdown_source ="""
                ###### Video Cache:
                By default, the viewer loads each frame sequentially. 

                You can use the button below to load the entire video into VRAM. This will enable smooth playback, but may require a lot of VRAM and loading time (i.e., 5 minutes for a 5 second video).
                """
                server.gui.add_markdown(markdown_source)
                load_video_button = self.server.gui.add_button(
                    "Load Video Cache",
                    hint="Load the entire video into VRAM to enable smooth playback."
                )
                @load_video_button.on_click
                def _(_) -> None:
                    self._video_cache = self.load_video_callback(CURRENT_PATH)
                    self._num_timesteps = self._video_cache["means"].shape[0]
                    self.frame_slider.max = self._num_timesteps - 1
                    self.rerender(_)

                start_stop_video_button = self.server.gui.add_button(
                    "Start/Stop",
                    hint="Start or stop video playback."
                )
                @start_stop_video_button.on_click
                def _(_) -> None:
                    if self.render_tab_state.video_stopped:
                        self._video_pause_event.set()
                        self.render_tab_state.video_stopped = False
                    else:
                        self._video_pause_event.clear()
                        self.render_tab_state.video_stopped = True

                fps_slider = server.gui.add_slider(
                    "FPS",
                    initial_value=self.render_tab_state.fps,
                    min=1,
                    max=60,
                    step=1,
                    hint="Frames per second for video playback.",
                )

                @fps_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.fps = fps_slider.value
                    self.rerender(_)

                self.frame_slider = server.gui.add_slider(
                    "Frame",
                    initial_value=self._timestep,
                    min=0,
                    max=self._num_timesteps - 1,
                    step=1,
                    hint="Current frame for video playback.",
                )

                @self.frame_slider.on_update
                def _(_) -> None:
                    self._timestep = self.frame_slider.value
                    if self.render_tab_state.video_stopped :
                        self.load_callback(CURRENT_PATH, self.render_tab_state, timestep=self._timestep, video_cache=self._video_cache)
                        self.rerender(_)
                    self.rerender(_)


        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
                "heatmap_range_vec2": heatmap_range_vec2,
                "max_scale_slider": max_scale_slider,
                "langsplat_text_input": langsplat_text_input,
                "fps_slider": fps_slider,
                "frame_slider": self.frame_slider,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count
