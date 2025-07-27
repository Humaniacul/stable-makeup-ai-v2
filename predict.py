import os
import tempfile
from typing import List
import torch
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe Face Mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Model setup complete - MediaPipe face detection and mesh initialized")
        
    def predict(
        self,
        source_image: Path = Input(description="Source face image"),
        reference_image: Path = Input(description="Reference makeup image"),
        makeup_intensity: float = Input(
            description="Makeup transfer intensity",
            default=1.0,
            ge=0.1,
            le=2.0,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        try:
            # Load images
            source_img = Image.open(source_image).convert("RGB")
            reference_img = Image.open(reference_image).convert("RGB")
            
            # Resize images to reasonable size
            source_img = source_img.resize((512, 512))
            reference_img = reference_img.resize((512, 512))
            
            # Convert to OpenCV format
            source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            reference_cv = cv2.cvtColor(np.array(reference_img), cv2.COLOR_RGB2BGR)
            
            # Apply makeup transfer
            result_cv = self.apply_makeup_transfer(source_cv, reference_cv, makeup_intensity)
            
            # Convert back to PIL Image
            result_img = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            
            # Save to temporary file with absolute path
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                output_path = tmp_file.name
                result_img.save(output_path, 'JPEG')
            
            return Path(output_path)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            # Return the source image as fallback
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                output_path = tmp_file.name
                source_img.save(output_path, 'JPEG')
            return Path(output_path)
    
    def apply_makeup_transfer(self, source: np.ndarray, reference: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply makeup transfer from reference to source image
        This is a simplified implementation using color transfer and blending
        """
        
        try:
            # Detect faces in both images
            source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
            
            # Get face landmarks
            source_landmarks = self.get_face_landmarks(source_rgb)
            reference_landmarks = self.get_face_landmarks(reference_rgb)
            
            if source_landmarks is None or reference_landmarks is None:
                print("Could not detect face landmarks, applying global color transfer")
                return self.apply_color_transfer(source, reference, intensity)
            
            # Apply makeup to specific facial regions
            result = source.copy()
            
            # Apply color transfer to facial regions
            face_mask = self.create_face_mask(source, source_landmarks)
            if face_mask is not None:
                # Apply color transfer within face region
                face_region = cv2.bitwise_and(source, source, mask=face_mask)
                ref_region = cv2.bitwise_and(reference, reference, mask=face_mask)
                
                transferred = self.apply_color_transfer(face_region, ref_region, intensity)
                
                # Blend with original
                face_mask_3d = cv2.merge([face_mask, face_mask, face_mask])
                face_mask_3d = face_mask_3d.astype(float) / 255.0
                
                result = result.astype(float)
                transferred = transferred.astype(float)
                
                result = result * (1 - face_mask_3d * intensity) + transferred * face_mask_3d * intensity
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"Error in makeup transfer: {e}")
            return self.apply_color_transfer(source, reference, intensity)
    
    def get_face_landmarks(self, image: np.ndarray):
        """Get facial landmarks using MediaPipe"""
        
        try:
            results = self.face_mesh.process(image)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
            return None
        except Exception as e:
            print(f"Error getting face landmarks: {e}")
            return None
    
    def create_face_mask(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Create a face mask from landmarks"""
        
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get face contour points
            face_points = []
            face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            for idx in face_oval_indices:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append([x, y])
            
            # Create convex hull and fill
            face_points = np.array(face_points)
            hull = cv2.convexHull(face_points)
            cv2.fillPoly(mask, [hull], 255)
            
            # Smooth the mask
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            return mask
            
        except Exception as e:
            print(f"Error creating face mask: {e}")
            return None
    
    def apply_color_transfer(self, source: np.ndarray, reference: np.ndarray, intensity: float) -> np.ndarray:
        """Apply color transfer from reference to source image"""
        
        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Calculate mean and std for each channel
            source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
            source_std = np.std(source_lab.reshape(-1, 3), axis=0)
            
            reference_mean = np.mean(reference_lab.reshape(-1, 3), axis=0)
            reference_std = np.std(reference_lab.reshape(-1, 3), axis=0)
            
            # Apply color transfer
            result_lab = source_lab.copy()
            for i in range(3):
                if source_std[i] > 0:
                    result_lab[:, :, i] = (result_lab[:, :, i] - source_mean[i]) * (reference_std[i] / source_std[i]) + reference_mean[i]
            
            # Blend with original based on intensity
            result_lab = source_lab * (1 - intensity) + result_lab * intensity
            
            # Convert back to BGR
            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            print(f"Error in color transfer: {e}")
            return source 