"""
convert_to_onnx.py
---
Convert Keras model to ONNX format for lightweight deployment on Vercel.
ONNX Runtime (~50MB) vs TensorFlow (~2GB) - 40x smaller!

Usage:
    python convert_to_onnx.py
"""

import os
import tensorflow as tf

def convert_model():
    """Convert mask_detector.keras to ONNX format."""
    keras_model_path = "model/mask_detector.keras"
    onnx_model_path = "model/mask_detector.onnx"
    
    if not os.path.exists(keras_model_path):
        print(f"❌ Model not found at {keras_model_path}")
        return False
    
    try:
        print("📦 Loading Keras model...")
        model = tf.keras.models.load_model(keras_model_path)
        
        print("🔄 Converting to ONNX format...")
        import tf2onnx
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)
        output_path = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)
        
        print(f"✅ ONNX model saved to {onnx_model_path}")
        print(f"📊 File size: {os.path.getsize(onnx_model_path) / 1024 / 1024:.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_model()
