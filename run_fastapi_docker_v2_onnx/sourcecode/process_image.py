import os
import shutil
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

# Đường dẫn mô hình
MODEL_PATH = 'model_repository/instruct-pix2pix'
INPUT_IMAGES_DIR = 'input_images'
OUTPUT_IMAGES_DIR = 'output_images'

# Load mô hình ONNX (UNet, VAE, Text Encoder)
unet_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'unet', 'unet.onnx'))
vae_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'vae', 'vae.onnx'))
text_encoder_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'text_encoder', 'text_encoder.onnx'))

# Hàm xử lý ảnh
def process_image(image: UploadFile, prompt: str):
    # Lưu ảnh đầu vào
    input_image_path = os.path.join(INPUT_IMAGES_DIR, image.filename)
    with open(input_image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Chuyển ảnh thành tensor (Ví dụ sử dụng PIL và numpy)
    img = Image.open(input_image_path).convert("RGB")
    img = np.array(img)
    img = img.astype(np.float32)

    # Giải mã và xử lý với UNet và VAE
    unet_input = np.expand_dims(img, axis=0)  # Dummy tensor cho UNet
    unet_output = unet_session.run(None, {'input': unet_input})

    # Quá trình với VAE và text encoder tương tự
    vae_output = vae_session.run(None, {'input': unet_output[0]})
    text_input = np.array([prompt], dtype=np.str)  # Đưa prompt vào text_encoder
    text_output = text_encoder_session.run(None, {'input': text_input})

    # Lưu ảnh xử lý
    output_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"processed_{image.filename}")
    Image.fromarray(vae_output[0].astype(np.uint8)).save(output_image_path)

    return output_image_path
