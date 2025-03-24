########load denoising
from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer
import logging
from skimage import exposure
from scipy.ndimage import gaussian_filter, uniform_filter

app = FastAPI()
triton_client = httpclient.InferenceServerClient("localhost:8000")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_image(image):
    """Chuyển ảnh về phạm vi [-1, 1]"""
    return (image - 0.5) * 2

@app.post("/predict/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    """Nhận ảnh và prompt, xử lý với Triton Server theo Stable Diffusion"""
    
    # Mã hóa văn bản
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
    text_embedding = text_embedding[:, :5, :] if text_embedding.shape[1] > 5 else np.pad(text_embedding, ((0, 0), (0, 5 - text_embedding.shape[1]), (0, 0)), mode='constant')
    
    logger.info(f"Text embedding shape: {text_embedding.shape}")
    
    # Xử lý ảnh
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512), Image.LANCZOS)
    image_np = np.transpose(np.array(image).astype(np.float32) / 255.0, (2, 0, 1))[None, :, :, :]
    image_np = normalize_image(image_np)
    
    # Mã hóa ảnh với VAE Encoder
    vae_response = triton_client.infer(
        "vae_encoder",
        inputs=[httpclient.InferInput("image", image_np.shape, "FP32").set_data_from_numpy(image_np)],
        outputs=[httpclient.InferRequestedOutput("latent")]
    )
    latent = vae_response.as_numpy("latent").astype(np.float32)

    logger.info(f"Latent shape from VAE Encoder: {latent.shape}")

    # **Vòng lặp denoising**
    num_inference_steps = 10 # Số bước khử nhiễu
    for step in range(num_inference_steps):
        timestep = np.array([1000 - step * (1000 // num_inference_steps)], dtype=np.float32)  # Thời gian bước giảm dần

        unet_response = triton_client.infer(
            "unet",
            inputs=[
                httpclient.InferInput("latents", latent.shape, "FP16").set_data_from_numpy(latent.astype(np.float16)),
                httpclient.InferInput("timestep", (1,), "FP16").set_data_from_numpy(timestep.astype(np.float16)),
                httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16").set_data_from_numpy(text_embedding.astype(np.float16))
            ],
            outputs=[httpclient.InferRequestedOutput("predicted_noise")]
        )
        
        predicted_noise = unet_response.as_numpy("predicted_noise").astype(np.float32)

        # **🔴 Xử lý lỗi số kênh không khớp giữa latent và predicted_noise**
        if predicted_noise.shape[1] == 4 and latent.shape[1] == 8:
            predicted_noise = np.concatenate([predicted_noise, predicted_noise], axis=1)

        logger.info(f"Predicted noise shape: {predicted_noise.shape}")

        # Cập nhật latent
        latent = latent - (predicted_noise * (1 / num_inference_steps))

    logger.info(f"Denoised latent shape: {latent.shape}, min: {latent.min()}, max: {latent.max()}")

    # **Chỉ lấy 4 kênh đầu tiên trước khi đưa vào VAE Decoder**
    latent = latent[:, :4, :, :]
    
    # Decode latent space thành ảnh
    vae_dec_response = triton_client.infer(
        "vae_decoder",
        inputs=[httpclient.InferInput("latent", latent.shape, "FP32").set_data_from_numpy(latent)],
        outputs=[httpclient.InferRequestedOutput("image")]
    )
    
    generated_image = vae_dec_response.as_numpy("image")[0]
    final_image = np.clip((generated_image + 1) / 2 * 255, 0, 255).astype(np.uint8)
    final_image = np.transpose(final_image, (1, 2, 0))
    
    # Hậu xử lý ảnh để tăng độ nét và giảm nhiễu
    final_image = exposure.adjust_gamma(final_image, gamma=0.85)  # Giữ sáng tự nhiên
    final_image = exposure.adjust_sigmoid(final_image, cutoff=0.5, gain=2.5)  # Cải thiện chi tiết
    final_image = gaussian_filter(final_image, sigma=0.8)  # Làm mượt nhẹ
    final_image = uniform_filter(final_image, size=1)  # Loại bỏ nhiễu nhỏ
    
    # Xuất ảnh
    pil_img = Image.fromarray((final_image * 255).astype(np.uint8))
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)
    
    return Response(img_io.getvalue(), media_type="image/png")