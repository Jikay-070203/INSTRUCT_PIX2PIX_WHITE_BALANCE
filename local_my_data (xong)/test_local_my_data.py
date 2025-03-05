import PIL
import torch
import os
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps

# Load mô hình
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Thư mục input và output
INPUT_DIR = "D:\\SourceCode\\ProjectOJT\\complete\\OJT_TASK3_LOCAL\\Deploy\\result\\output_resize"
OUTPUT_DIR = "D:\\SourceCode\\ProjectOJT\\complete\\OJT_TASK3_LOCAL\\Deploy\\result\\output_image_pix_model"

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_image(image_path, size=(512, 512)):
    """Resize ảnh về kích thước chuẩn."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)  # Xử lý xoay ảnh
    image = image.convert("RGB")
    image = image.resize(size, Image.LANCZOS)
    return image

def brighten_image(image_path):
    """Làm sáng ảnh bằng instruct-pix2pix."""
    # Resize ảnh về 512x512
    image = resize_image(image_path)

    # Chạy mô hình với prompt 
    prompt = "photo white balance"
    result = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]

    # Lưu ảnh với tên mới
    filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_DIR, filename.replace(".png", "_brightened.jpg"))
    result.save(output_path)
    print(f"✅ Ảnh đã được xử lý và lưu tại: {output_path}")

# Chạy trên một ảnh cụ thể
image_path = os.path.join(INPUT_DIR, "IMG_1684.png")  # iamge cần white balance
brighten_image(image_path)
