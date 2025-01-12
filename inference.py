import os
import sys
import shutil
import torch
from time import strftime
from argparse import ArgumentParser
from glob import glob

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


def safe_mkdir(directory):
    """Create a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)


def validate_paths(paths):
    """Validate that required paths exist."""
    for path in paths:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")


def setup_logging(log_dir):
    """Setup logging directory."""
    safe_mkdir(log_dir)
    log_file = os.path.join(log_dir, 'process.log')
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout


def main(args):
    # Logging setup
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    setup_logging(save_dir)

    # Validate input paths
    validate_paths([args.source_image, args.driven_audio, args.checkpoint_dir])

    # Prepare directories
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    safe_mkdir(first_frame_dir)

    print("Initializing paths and models...")
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(os.path.dirname(__file__), 'src/config'),
                                args.size, args.old_version, args.preprocess)

    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)

    # 3DMM Extraction for source image
    print("Extracting 3DMM for source image...")
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        args.source_image, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size
    )

    if first_coeff_path is None:
        print("Error: Unable to extract coefficients from the source image.")
        return

    # Process reference videos for eye blinking and pose
    ref_eyeblink_coeff_path = None
    if args.ref_eyeblink:
        ref_eyeblink_dir = os.path.join(save_dir, 'ref_eyeblink')
        safe_mkdir(ref_eyeblink_dir)
        print("Extracting 3DMM for reference video (eye blinking)...")
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            args.ref_eyeblink, ref_eyeblink_dir, args.preprocess, source_image_flag=False
        )

    ref_pose_coeff_path = None
    if args.ref_pose:
        ref_pose_dir = os.path.join(save_dir, 'ref_pose')
        safe_mkdir(ref_pose_dir)
        print("Extracting 3DMM for reference video (pose)...")
        ref_pose_coeff_path, _, _ = preprocess_model.generate(
            args.ref_pose, ref_pose_dir, args.preprocess, source_image_flag=False
        )

    # Generate coefficients from audio
    print("Generating coefficients from audio...")
    batch = get_data(first_coeff_path, args.driven_audio, args.device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, ref_pose_coeff_path)

    # Generate 3D face visualization (optional)
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        print("Generating 3D face visualization...")
        gen_composed_video(args, args.device, first_coeff_path, coeff_path, args.driven_audio,
                           os.path.join(save_dir, '3dface.mp4'))

    # Generate the final video
    print("Generating final animated video...")
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, args.driven_audio,
                               args.batch_size, args.input_yaw, args.input_pitch, args.input_roll,
                               expression_scale=args.expression_scale, still_mode=args.still,
                               preprocess=args.preprocess, size=args.size)

    result_video = animate_from_coeff.generate(
        data, save_dir, args.source_image, crop_info, enhancer=args.enhancer,
        background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size
    )

    shutil.move(result_video, f"{save_dir}.mp4")
    print(f"The generated video is saved at: {save_dir}.mp4")

    # Clean up intermediate files if verbose mode is off
    if not args.verbose:
        print("Cleaning up intermediate files...")
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--driven_audio", required=True, help="Path to driven audio")
    parser.add_argument("--source_image", required=True, help="Path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="Path to reference video for eye blinking")
    parser.add_argument("--ref_pose", default=None, help="Path to reference video for pose")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to model checkpoints")
    parser.add_argument("--result_dir", default="./results", help="Path to save results")
    parser.add_argument("--pose_style", type=int, default=0, help="Pose style index")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for face rendering")
    parser.add_argument("--size", type=int, default=256, help="Image size for processing")
    parser.add_argument("--expression_scale", type=float, default=1.0, help="Expression intensity scale")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="Input yaw angles")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="Input pitch angles")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="Input roll angles")
    parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer (e.g., gfpgan)")
    parser.add_argument('--background_enhancer', type=str, default=None, help="Background enhancer (e.g., realesrgan)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--face3dvis", action="store_true", help="Generate 3D face visualization")
    parser.add_argument("--still", action="store_true", help="Enable still mode for full body animation")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'],
                        help="Preprocessing method")
    parser.add_argument("--verbose", action="store_true", help="Keep intermediate files")
    parser.add_argument("--old_version", action="store_true", help="Use old checkpoint version")

    args = parser.parse_args()

    # Set device
    args.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Run main process
    main(args)