import argparse
import logging
from config import configure_project
from implementations.create_depth_maps import CreateDepthMaps
from implementations.light_gaussian import LightGaussian
from implementations.dynamic_gaussian import DynamicGaussian
from implementations.original_gaussian_splatting import OriginalGaussianSplatting
from implementations.viewer import Viewer
from implementations.langsplat import LangSplat
from implementations.endogaussian import EndoGaussian

logger = logging.getLogger(__name__)


def main():
    configure_project()

    parser = argparse.ArgumentParser(
        description="Run different implementations of Gaussian Splatting."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="implementation", required=True)

    # Subparser for 'original'
    original_parser = subparsers.add_parser(
        "original", help="Run the original implementation."
    )
    original_parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode for the original implementation.",
    )
    original_parser.add_argument(
        "--use_depth",
        action="store_true",
        help="Use depth in the original implementation.",
    )
    original_parser.add_argument(
        "--use_expcomp",
        action="store_true",
        help="Use exposure compensation in the original implementation.",
    )
    original_parser.add_argument(
        "--use_viewer",
        action="store_true",
        help="Use viewer during training.",
    )

    depth_maps_parser = subparsers.add_parser(
        "create_depth_maps", help="Create depth maps for the datasets."
    )

    # Subparser for 'LightGaussian'
    light_gaussian_parser = subparsers.add_parser(
        "LightGaussian", help="Run the Light Gaussian implementation."
    )
    light_gaussian_parser.add_argument(
        "--use_viewer", action="store_true", help="Use viewer during training."
    )

    # Subparser for 'Viewer'
    viewer_parser = subparsers.add_parser("viewer", help="Run the viewer.")
    viewer_parser.add_argument(
        "--path", type=str, help="Path to the data for the viewer."
    )

    # Subparser for 'LangSplat'
    langsplat_parser = subparsers.add_parser(
        "LangSplat", help="Run the LangSplat implementation."
    )
    langsplat_parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run the entire LangSplat pipeline. Use --resolutions [INT] to additionaly specify the resolution.",
    )
    langsplat_parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess the inputs using SAM and CLIP.",
    )
    langsplat_parser.add_argument(
        "--train_autoencoder",
        action="store_true",
        help="Train autoencoder to map to a smaller latent space.",
    )
    langsplat_parser.add_argument(
        "--train_3dgs",
        action="store_true",
        help="Train original 3DGS to get the checkpoint needed for LangSplat training. ",
    )
    langsplat_parser.add_argument(
        "--gs_impl",
        nargs="?",
        const=True,
        type=str,
        default="original",
        help="Implementation of Gaussian Splatting to use for LangSplat training. Options: 'original', 'LightGaussian'.",
    )
    langsplat_parser.add_argument(
        "--train_semantic_features",
        action="store_true",
        help="Train 3D Gaussians to have semantic features.",
    )
    langsplat_parser.add_argument(
        "--resolution",
        nargs="?",
        const=True,
        type=int,
        default=None,
        help="Value by which the image resolution is being divided in the preprocessing, 3DGS training and LangSplat training steps. The higher the value, the shorter the training time.",
    )
    langsplat_parser.add_argument(
        "--scenario",
        nargs="?",
        const=True,
        type=str,
        default=None,
        help="Scenario name to process.",
    )
    langsplat_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the LangSplat after training.",
    )

    # Subparser for 'DynamicGaussian'
    dynamic_gaussian_parser = subparsers.add_parser(
        "DynamicGaussian", help="Run the Dynamic3DGaussians implementation."
    )
    dynamic_gaussian_parser.add_argument(
        "--input_path",
        nargs="?",
        const=True,
        type=str,
        default=None,
        help="Path to input dataset. It is also possible to specify a path to a folder that contains multiple dynamic gaussian datasets.",
    )
    dynamic_gaussian_parser.add_argument(
        "--output_path",
        nargs="?",
        const=True,
        type=str,
        default=None,
        help="Path to output directory. If not specified, the default output path from the repo will be used.",
    )

    # Subparser for 'EndoGaussian'
    endo_gaussian_parser = subparsers.add_parser(
        "EndoGaussian", help="Run the EndoGaussian implementation."
    )
    endo_gaussian_parser.add_argument(
        "--pulling",
        action="store_true",
        help="Enable pulling in the EndoGaussian implementation.",
    )
    endo_gaussian_parser.add_argument(
        "--cutting",
        action="store_true",
        help="Enable cutting in the EndoGaussian implementation.",
    )

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.debug("Debug mode is enabled.")

    if args.implementation == "original":
        gs_impl = OriginalGaussianSplatting(
            fast=args.fast,
            use_depth=args.use_depth,
            use_expcomp=args.use_expcomp,
            use_viewer=args.use_viewer,
        )
        gs_impl.execute()
    elif args.implementation == "LightGaussian":
        light_gaussian = LightGaussian(use_viewer=args.use_viewer)
        light_gaussian.execute()
    elif args.implementation == "LangSplat":
        langsplat = LangSplat(
            run_all=True if args.scenario else args.run_all,
            preprocess=args.preprocess,
            train_autoencoder=args.train_autoencoder,
            train_3dgs=args.train_3dgs,
            train_semantic_features=args.train_semantic_features,
            resolution=args.resolution,
            scenario=args.scenario,
            gs_impl=args.gs_impl,
            render=args.render,
        )
        langsplat.execute()
    elif args.implementation == "EndoGaussian":
        endo_gaussian = EndoGaussian(
            pulling=args.pulling,
            cutting=args.cutting
        )
        endo_gaussian.execute()
    elif args.implementation == "DynamicGaussian":
        dynamic_gaussian = DynamicGaussian(
            input_path=args.input_path,
            output_path=args.output_path
        )
        dynamic_gaussian.execute()
    elif args.implementation == "create_depth_maps":
        create_depth_maps = CreateDepthMaps()
        create_depth_maps.execute()
    elif args.implementation == "viewer":
        viewer = Viewer()
        viewer.execute(args.path)
    else:
        raise ValueError(f"Unknown implementation: {args.implementation}")


if __name__ == "__main__":
    main()
