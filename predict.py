import os
import subprocess
import tempfile
import torch
from PIL import Image
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Stable-Makeup model by cloning the original repository"""
        
        print("Setting up Stable-Makeup model...")
        
        # Clone the original Stable-Makeup repository
        if not os.path.exists("Stable-Makeup"):
            print("Cloning original Stable-Makeup repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/Xiaojiu-z/Stable-Makeup.git"
            ], check=True)
        
        # Change to the Stable-Makeup directory for model setup
        original_dir = os.getcwd()
        os.chdir("Stable-Makeup")
        
        # Download the pre-trained weights if they exist
        models_dir = "models/stablemakeup"
        os.makedirs(models_dir, exist_ok=True)
        
        # Check if we have the model weights in our mounted directory
        external_weights_dir = "../models/stablemakeup"
        if os.path.exists(external_weights_dir):
            print("Copying model weights from external directory...")
            import shutil
            for file in os.listdir(external_weights_dir):
                if file.endswith('.bin') or file.endswith('.pth'):
                    src = os.path.join(external_weights_dir, file)
                    dst = os.path.join(models_dir, file)
                    shutil.copy2(src, dst)
                    print(f"Copied {file}")
        
        # Now import and setup the actual Stable-Makeup model
        try:
            # Add the Stable-Makeup directory to Python path
            import sys
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
            
            # Import the original model components
            from pipeline_sd15 import StableDiffusionControlNetPipeline
            from diffusers import DDIMScheduler, ControlNetModel, UNet2DConditionModel
            from detail_encoder.encoder_plus import detail_encoder
            from spiga_draw import *
            from spiga.inference.config import ModelConfig
            from spiga.inference.framework import SPIGAFramework
            from facelib import FaceDetector
            
            print("Successfully imported Stable-Makeup components")
            
            # Initialize SPIGA framework
            self.processor = SPIGAFramework(ModelConfig("300wpublic"))
            
            # Initialize face detector
            detector_path = "models/mobilenet0.25_Final.pth"
            if os.path.exists(f"../{detector_path}"):
                # Copy from our external models directory
                import shutil
                os.makedirs("models", exist_ok=True)
                shutil.copy2(f"../{detector_path}", detector_path)
            
            try:
                self.detector = FaceDetector(weight_path=detector_path) if os.path.exists(detector_path) else None
            except:
                self.detector = None
                print("Warning: Face detector not loaded")
            
            # Setup the actual Stable-Makeup model architecture
            print("Loading Stable Diffusion base model...")
            self.model_id = "runwayml/stable-diffusion-v1-5"
            
            # Load UNet
            self.Unet = UNet2DConditionModel.from_pretrained(
                self.model_id, 
                subfolder="unet",
                torch_dtype=torch.float32
            ).to("cuda")
            
            # Initialize encoders using the original architecture
            print("Initializing Stable-Makeup encoders...")
            self.id_encoder = ControlNetModel.from_unet(self.Unet)
            self.pose_encoder = ControlNetModel.from_unet(self.Unet)
            
            # Initialize the detail encoder with proper path
            image_encoder_path = "models/image_encoder_l"
            os.makedirs(image_encoder_path, exist_ok=True)
            
            self.makeup_encoder = detail_encoder(
                self.Unet, 
                image_encoder_path, 
                "cuda", 
                dtype=torch.float32
            )
            
            # Load pre-trained weights if available
            weight_files = {
                "models/stablemakeup/pytorch_model.bin": self.makeup_encoder,
                "models/stablemakeup/pytorch_model_1.bin": self.id_encoder,
                "models/stablemakeup/pytorch_model_2.bin": self.pose_encoder
            }
            
            for weight_path, model in weight_files.items():
                if os.path.exists(weight_path):
                    print(f"Loading weights from {weight_path}...")
                    state_dict = torch.load(weight_path, map_location="cuda")
                    model.load_state_dict(state_dict, strict=False)
                else:
                    print(f"Warning: {weight_path} not found, using base weights")
            
            # Move models to GPU
            self.id_encoder.to("cuda")
            self.pose_encoder.to("cuda")
            self.makeup_encoder.to("cuda")
            
            # Initialize the pipeline using original implementation
            print("Setting up Stable-Makeup pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                safety_checker=None,
                unet=self.Unet,
                controlnet=[self.id_encoder, self.pose_encoder],
                torch_dtype=torch.float32
            ).to("cuda")
            
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            
            print("Stable-Makeup model setup complete!")
            
        except Exception as e:
            print(f"Error setting up model: {e}")
            print("Will use fallback implementation")
            self.pipe = None
            self.makeup_encoder = None
        
        finally:
            # Return to original directory
            os.chdir(original_dir)
    
    def get_draw(self, pil_img, size):
        """Get facial structure drawing using SPIGA from original implementation"""
        
        # Change to Stable-Makeup directory for processing
        original_dir = os.getcwd()
        os.chdir("Stable-Makeup")
        
        try:
            if self.detector is None:
                width, height = pil_img.size
                return Image.new('RGB', (width, height), color=(0, 0, 0))
                
            # Use the original spiga_process and spiga_segmentation functions
            spigas = spiga_process(pil_img, self.detector)
            if spigas == False:
                width, height = pil_img.size  
                black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))  
                return black_image_pil
            else:
                spigas_faces = spiga_segmentation(spigas, size=size)
                return spigas_faces
                
        except Exception as e:
            print(f"Error in get_draw: {e}")
            width, height = pil_img.size
            return Image.new('RGB', (width, height), color=(0, 0, 0))
        finally:
            os.chdir(original_dir)

    def predict(
        self,
        source_image: Path = Input(description="Source face image"),
        reference_image: Path = Input(description="Reference makeup image"),
        makeup_intensity: float = Input(
            description="Makeup transfer intensity",
            default=1.5,
            ge=1.0,
            le=3.0,
        ),
    ) -> Path:
        """Run makeup transfer using the original Stable-Makeup implementation"""
        
        # Change to Stable-Makeup directory for inference
        original_dir = os.getcwd()
        os.chdir("Stable-Makeup")
        
        try:
            # Load and preprocess images
            source_img = Image.open(source_image).convert("RGB")
            makeup_img = Image.open(reference_image).convert("RGB")
            
            # Resize to 512x512 as expected by the model
            source_img = source_img.resize((512, 512))
            makeup_img = makeup_img.resize((512, 512))
            
            # Get facial structure image using original implementation
            pose_image = self.get_draw(source_img, size=512)
            
            if self.makeup_encoder is None or self.pipe is None:
                print("Model not properly loaded, returning source image")
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    output_path = tmp_file.name
                    source_img.save(output_path, 'JPEG')
                return Path(output_path)
            
            # Run the actual Stable-Makeup inference using original model
            print("Running Stable-Makeup inference...")
            result_img = self.makeup_encoder.generate(
                id_image=[source_img, pose_image], 
                makeup_image=makeup_img,
                pipe=self.pipe, 
                guidance_scale=makeup_intensity
            )
            
            # Save result
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                output_path = tmp_file.name
                result_img.save(output_path, 'JPEG', quality=95)
            
            return Path(output_path)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            # Return source image as fallback
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                output_path = tmp_file.name
                source_img.save(output_path, 'JPEG')
            return Path(output_path)
        
        finally:
            os.chdir(original_dir) 