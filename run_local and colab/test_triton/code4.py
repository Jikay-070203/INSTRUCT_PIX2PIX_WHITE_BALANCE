from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer
import logging
from skimage import exposure
from scipy.ndimage import gaussian_filter

app = FastAPI()
triton_client = httpclient.InferenceServerClient("localhost:8000")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    """Nhận ảnh và prompt, xử lý với Triton Server"""

    # Mã hóa văn bản với Text Encoder
    tokenized = tokenizer(prompt, padding="max_length", max_length=7, return_tensors="np")
    tokens = tokenized["input_ids"].astype(np.int64)
    attention_mask = tokenized["attention_mask"].astype(np.int64)  

    # Gửi request đến text_encoder
    text_response = triton_client.infer(
        "text_encoder",
        inputs=[
            httpclient.InferInput("input_text", tokens.shape, "INT64").set_data_from_numpy(tokens),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64").set_data_from_numpy(attention_mask)
        ],
        outputs=[httpclient.InferRequestedOutput("text_embeddings")]
    )

    text_embedding = text_response.as_numpy("text_embeddings").astype(np.float32)
    
    # Điều chỉnh shape về [N,5,768] nếu cần
    if text_embedding.shape[1] > 5:
        text_embedding = text_embedding[:, :5, :]
    elif text_embedding.shape[1] < 5:
        padding = np.zeros((text_embedding.shape[0], 5 - text_embedding.shape[1], text_embedding.shape[2]), dtype=np.float32)
        text_embedding = np.concatenate((text_embedding, padding), axis=1)

    logger.info(f"Text embedding shape: {text_embedding.shape}")

    # Mã hóa ảnh với VAE Encoder
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512), Image.LANCZOS)
    image_np = np.transpose(np.array(image).astype(np.float32) / 255.0, (2, 0, 1))[None, :, :, :]

    vae_response = triton_client.infer(
        "vae_encoder",
        inputs=[httpclient.InferInput("image", image_np.shape, "FP32").set_data_from_numpy(image_np)],
        outputs=[httpclient.InferRequestedOutput("latent")]
    )
    latent = vae_response.as_numpy("latent").astype(np.float32)
    logger.info(f"Latent shape: {latent.shape}, min: {latent.min()}, max: {latent.max()}")

    # Gửi dữ liệu vào UNet để xử lý nhiễu
    unet_response = triton_client.infer(
        "unet",
        inputs=[
            httpclient.InferInput("latents", latent.shape, "FP16").set_data_from_numpy(latent.astype(np.float16)),
            httpclient.InferInput("timestep", (1,), "FP16").set_data_from_numpy(np.array([1], dtype=np.float16)),
            httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16").set_data_from_numpy(text_embedding.astype(np.float16))
        ],
        outputs=[httpclient.InferRequestedOutput("predicted_noise")]
    )

    # 🔹 **Điều chỉnh hệ số để giữ nhiều chi tiết hơn**
    denoised_latents = unet_response.as_numpy("predicted_noise").astype(np.float32) * 1.02  
    logger.info(f"Denoised latents shape: {denoised_latents.shape}, min: {denoised_latents.min()}, max: {denoised_latents.max()}")

    # Decode latent space thành ảnh
    vae_dec_response = triton_client.infer(
        "vae_decoder",
        inputs=[httpclient.InferInput("latent", denoised_latents.shape, "FP32").set_data_from_numpy(denoised_latents)],
        outputs=[httpclient.InferRequestedOutput("image")]
    )

    generated_image = vae_dec_response.as_numpy("image")

    # 📌 Kiểm tra lỗi nếu không có dữ liệu trả về từ VAE Decoder
    if generated_image is None or generated_image.size == 0:
        raise ValueError("Lỗi: Không nhận được dữ liệu hợp lệ từ vae_decoder!")

    generated_image = generated_image[0]  # Lấy ảnh từ batch đầu tiên nếu có batch
    logger.info(f"Bước thử 5.1: Giá trị trước chuẩn hóa - min: {generated_image.min()}, max: {generated_image.max()}")

    # Đưa ảnh về khoảng [0,1] trước khi chuyển sang uint8
    final_image = np.clip((generated_image + 1) / 2, 0, 1) * 255
    final_image = final_image.astype(np.uint8)

    # Kiểm tra shape & điều chỉnh nếu cần
    if final_image.shape[0] == 1:
        final_image = np.repeat(final_image, 3, axis=0)  # Tạo RGB từ grayscale
    elif final_image.shape[0] == 4:
        final_image = final_image[:3, :, :]  # Bỏ kênh alpha nếu có

    # Chuyển từ (C, H, W) → (H, W, C)
    final_image = np.transpose(final_image, (1, 2, 0))

    # 🔹 **Sử dụng Adaptive Histogram Equalization (CLAHE) trên từng kênh**
    final_image = np.stack([exposure.equalize_adapthist(final_image[:, :, i]) for i in range(3)], axis=-1)

    # 🔹 **Giảm sáng để tránh mất cân bằng màu**
    final_image = exposure.adjust_gamma(final_image, gamma=0.95)  # Giảm sáng nhẹ

    # 🔹 **Tăng chi tiết nhưng không làm rực màu**
    final_image = exposure.adjust_sigmoid(final_image, cutoff=0.5, gain=2.5)  # Giữ màu tự nhiên hơn

    # 🔹 **Làm mượt nhẹ để giảm nhiễu**
    final_image = gaussian_filter(final_image, sigma=0.2)

    # Chuyển về uint8 để lưu ảnh
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

    logger.info(f"Final generated_image shape: {final_image.shape}, min: {final_image.min()}, max: {final_image.max()}")

    # Xuất ảnh ra response
    pil_img = Image.fromarray(final_image)
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(img_io.getvalue(), media_type="image/png")
