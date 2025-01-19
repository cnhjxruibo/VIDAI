import os
import shutil
from argparse import Namespace
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from cog import BasePredictor, Input, Path

checkpoints = "checkpoints"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"
        self.sadtalker_paths = init_path(checkpoints, os.path.join("src", "config"))

        # Initialize models
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, device)
        self.animate_from_coeff = {
            "full": AnimateFromCoeff(self.sadtalker_paths, device),
            "others": AnimateFromCoeff(self.sadtalker_paths, device),
        }

    def predict(
        self,
        source_image: Path = Input(description="Upload the source image, it can be video.mp4 or picture.png"),
        driven_audio: Path = Input(description="Upload the driven audio, accepts .wav and .mp4 file"),
        enhancer: str = Input(
            description="Choose a face enhancer", choices=["gfpgan", "RestoreFormer"], default="gfpgan"
        ),
        preprocess: str = Input(
            description="How to preprocess the images", choices=["crop", "resize", "full"], default="full"
        ),
        ref_eyeblink: Path = Input(description="Path to reference video providing eye blinking", default=None),
        ref_pose: Path = Input(description="Path to reference video providing pose", default=None),
        still: bool = Input(
            description="Can crop back to the original videos for the full body animation when preprocess is full",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        def extract_coeff(path, output_dir, preprocess_flag):
            """Helper function to extract 3DMM coefficients"""
            os.makedirs(output_dir, exist_ok=True)
            return self.preprocess_model.generate(path, output_dir, preprocess_flag)

        def process_reference_video(video_path, results_dir, label):
            if video_path:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_dir = os.path.join(results_dir, f"{label}_{video_name}")
                return extract_coeff(video_path, frame_dir, preprocess)
            return None

        results_dir = "results"
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        # Process source image
        print("3DMM Extraction for source image")
        first_frame_dir = os.path.join(results_dir, "first_frame_dir")
        first_coeff_path, crop_pic_path, crop_info = extract_coeff(source_image, first_frame_dir, preprocess)

        if first_coeff_path is None:
            raise ValueError("Cannot extract coefficients from the source image.")

        # Process reference videos
        ref_eyeblink_coeff_path = process_reference_video(ref_eyeblink, results_dir, "ref_eyeblink")
        ref_pose_coeff_path = (
            ref_eyeblink_coeff_path if ref_pose == ref_eyeblink else process_reference_video(ref_pose, results_dir, "ref_pose")
        )

        # Audio to coefficient
        print("Audio to Coefficient")
        batch = get_data(
            first_coeff_path, str(driven_audio), "cuda", ref_eyeblink_coeff_path, still=still
        )
        coeff_path = self.audio_to_coeff.generate(batch, results_dir, pose_style=0, ref_pose_coeff_path=ref_pose_coeff_path)

        # Coefficient to video
        print("Coefficient to Video")
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            str(driven_audio),
            batch_size=2,
            input_yaw=None,
            input_pitch=None,
            input_roll=None,
            expression_scale=1.0,
            still_mode=still,
            preprocess=preprocess,
        )

        animator = self.animate_from_coeff["full" if preprocess == "full" else "others"]
        animator.generate(
            data,
            results_dir,
            str(source_image),
            crop_info,
            enhancer=enhancer,
            background_enhancer=None,
            preprocess=preprocess,
        )

        # Save final output
        output_path = "/tmp/out.mp4"
        enhanced_videos = [f for f in os.listdir(results_dir) if "enhanced.mp4" in f]
        if not enhanced_videos:
            raise FileNotFoundError("Enhanced video not found in results.")

        shutil.copy(os.path.join(results_dir, enhanced_videos[0]), output_path)
        return Path(output_path)


def load_default():
    """Load default arguments"""
    return Namespace(
        pose_style=0,
        batch_size=2,
        expression_scale=1.0,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        background_enhancer=None,
        face3dvis=False,
        net_recon="resnet50",
        init_path=None,
        use_last_fc=False,
        bfm_folder="./src/config/",
        bfm_model="BFM_model_front.mat",
        focal=1015.0,
        center=112.0,
        camera_d=10.0,
        z_near=5.0,
        z_far=15.0,
    )
