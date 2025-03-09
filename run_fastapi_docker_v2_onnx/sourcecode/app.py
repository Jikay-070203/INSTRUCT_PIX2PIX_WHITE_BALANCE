from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import shutil
import onnxruntime as ort
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F #thư viện này cần 

app = FastAPI()

#path cho cấu trúc ta cần 
MODEL_PATH = 'model_repository/instruct-pix2pix'
INPUT_IMAGES_DIR = 'input_images'
OUTPUT_IMAGES_DIR = 'output_images'

# Load mô hình ONNX
unet_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'unet', 'unet.onnx'))
vae_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'vae', 'vae.onnx'))
text_encoder_session = ort.InferenceSession(os.path.join(MODEL_PATH, 'text_encoder', 'text_encoder.onnx'))

# kiểm tra đầu vào
def debug_shape(name, tensor):
    print(f"{name} shape: {tensor.shape}")

# mã hóa prompt bằng text encoder
def encode_prompt(prompt: str):
    """Mã hóa prompt thành tensor dùng làm input cho UNet"""
    prompt_tokens = np.array([[ord(c) for c in prompt.ljust(77)]], dtype=np.int64)  
    encoded_prompt = text_encoder_session.run(None, {'input_ids': prompt_tokens})
    
    debug_shape("Encoded Prompt", encoded_prompt[0])  
    return encoded_prompt[0]

# resize ảnh đầu vào về 64x64
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((64, 64))  # Resize về đúng kích thước cho UNet
    img = np.array(img).astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]

    # Chuyển thành tensor có 8 kênh
    img = np.transpose(img, (2, 0, 1))  # Đưa về dạng (3, 64, 64)
    img = np.tile(img, (3, 1, 1))[:8]  # Nhân lên đủ 8 kênh

    return img[np.newaxis, :, :, :]  # Thêm batch dimension -> (1, 8, 64, 64)

# xử lý ảnh có truyền prompt
def process_image(image: UploadFile, prompt: str):
    input_image_path = os.path.join(INPUT_IMAGES_DIR, image.filename)
    with open(input_image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    sample_input = preprocess_image(input_image_path)
    timestep_input = np.array([1.0], dtype=np.float32)
    encoder_hidden_state_input = encode_prompt(prompt)  # Mã hóa prompt thay vì random tensor

    # test kích thước input
    debug_shape("Sample Input", sample_input)
    debug_shape("Timestep Input", timestep_input)
    debug_shape("Encoder Hidden State", encoder_hidden_state_input)

    # Chạy UNet với prompt 
    unet_output = unet_session.run(None, {
        'sample': sample_input,
        'timestep': timestep_input,
        'encoder_hidden_state': encoder_hidden_state_input  # Prompt 
    })
    
    debug_shape("UNet Output", unet_output[0])  # (1, 4, 64, 64)
    
    # Resize output về (1, 4, 512, 512)
    unet_output_tensor = torch.tensor(unet_output[0])
    unet_output_resized = F.interpolate(unet_output_tensor, size=(512, 512), mode="bilinear", align_corners=False).numpy()
    
    debug_shape("UNet Resized Output", unet_output_resized)  # (1, 4, 512, 512)
    
    # đưa 4 kênh -> 3 kênh trước khi đưa vào VAE
    unet_output_rgb = unet_output_resized[:, :3, :, :]
    debug_shape("Final Input to VAE", unet_output_rgb)  # (1, 3, 512, 512)

    # Chạy VAE để chuyển thành ảnh
    vae_output = vae_session.run(None, {'image': unet_output_rgb})
    debug_shape("VAE Output", vae_output[0])
    
    # Chuyển output về ảnh
    output_image = (vae_output[0] * 255).clip(0, 255).astype(np.uint8)
    output_image = np.transpose(output_image[0], (1, 2, 0))

    output_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"processed_{image.filename}")
    Image.fromarray(output_image).save(output_image_path)

    return output_image_path

# API nhận ảnh và prompt để xử lý
@app.post("/process-image/")
async def process_image_api(
    image: UploadFile = File(...),
    prompt: str = Form("photo white-balance")  # Mặc định nếu không nhập 
):
    output_image_path = process_image(image, prompt)
    return FileResponse(output_image_path)