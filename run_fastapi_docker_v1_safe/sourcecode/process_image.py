import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import os

# Định nghĩa đường dẫn model
MODEL_PATH = "model_repository/instruct-pix2pix"  # Nếu chạy local
if os.getenv("DOCKER_ENV"):  # Nếu chạy trong Docker
    MODEL_PATH = "/app/model"

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model từ thư mục
model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # Loại bỏ kiểm tra nếu cần
)
model.to(device)

def process_with_safetensors(image: Image.Image, model, prompt="Enhance image quality"):
    """Xử lý ảnh bằng mô hình instruct-pix2pix với prompt động"""
    image = image.convert("RGB")  # Đảm bảo ảnh đúng định dạng
    result = model(prompt=prompt, image=image).images[0]  # Gọi model với tham số đúng
    return result
