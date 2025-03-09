from fastapi import FastAPI, File, UploadFile, Form, Response
from PIL import Image
import io
import os
from process_image import process_with_safetensors, model  # Đảm bảo hàm xử lý có hỗ trợ prompt

app = FastAPI()

# Thư mục lưu ảnh đầu vào và đầu ra
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/enhance/")
async def enhance_image(
    file: UploadFile = File(...),
    prompt: str = Form("photo white-balance")  # Mặc định là "photo white-balance" nếu không nhập
):
    """API nhận ảnh + prompt, xử lý bằng mô hình và trả về ảnh kết quả."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return {"error": "File không hợp lệ"}

    # Lưu ảnh input trước khi xử lý
    input_path = os.path.join(INPUT_DIR, f"input_{file.filename}")
    image.save(input_path)

    # Gọi hàm xử lý ảnh có truyền prompt
    enhanced_image = process_with_safetensors(image, model, prompt)

    # Lưu ảnh output
    output_path = os.path.join(OUTPUT_DIR, f"enhanced_{file.filename}")
    enhanced_image.save(output_path)

    # Trả về ảnh output
    img_io = io.BytesIO()
    enhanced_image.save(img_io, format="PNG")
    img_io.seek(0)
    return Response(img_io.getvalue(), media_type="image/png")

@app.get("/")
async def root():
    return {"message": "API instruct-pix2pix is running"}
