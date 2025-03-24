# from transformers import CLIPTokenizer
# import os      
# model_path = r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\Deploy\WB\fastapi\model_repository\instruct-pix2pix\tokenizer"
# tokenizer = CLIPTokenizer.from_pretrained(model_path)
# print("Tokenizer loaded successfully!")


from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer
import logging
from skimage import exposure
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import os       

app = FastAPI()
triton_client = httpclient.InferenceServerClient("localhost:8000")
model_path = r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\Deploy\WB\fastapi\model_repository\instruct-pix2pix\tokenizer"
tokenizer = CLIPTokenizer.from_pretrained(model_path)

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
    
    logger.info(f"Latent shape: {latent.shape}, min: {latent.min()}, max: {latent.max()}")

    # Gửi dữ liệu vào UNet với timestep=10
    unet_response = triton_client.infer(
        "unet",
        inputs=[
            httpclient.InferInput("latents", latent.shape, "FP16").set_data_from_numpy(latent.astype(np.float16)),
            httpclient.InferInput("timestep", (1,), "FP16").set_data_from_numpy(np.array([10], dtype=np.float16)),
            httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16").set_data_from_numpy(text_embedding.astype(np.float16))
        ],
        outputs=[httpclient.InferRequestedOutput("predicted_noise")]
    )
    
    denoised_latents = unet_response.as_numpy("predicted_noise").astype(np.float32)

    # **Cải thiện cách xử lý latent trước khi đưa vào VAE Decoder**
    if denoised_latents.shape[1] > 4:
        denoised_latents = denoised_latents[:, :4, :, :]

    # **Làm mượt latent để tránh noise quá mức**
    denoised_latents = gaussian_filter(denoised_latents, sigma=0.5)
    
    logger.info(f"Denoised latent shape: {denoised_latents.shape}, min: {denoised_latents.min()}, max: {denoised_latents.max()}")

    # Decode latent space thành ảnh
    vae_dec_response = triton_client.infer(
        "vae_decoder",
        inputs=[httpclient.InferInput("latent", denoised_latents.shape, "FP32").set_data_from_numpy(denoised_latents)],
        outputs=[httpclient.InferRequestedOutput("image")]
    )
    
    generated_image = vae_dec_response.as_numpy("image")[0]
    final_image = np.clip((generated_image + 1) / 2 * 255, 0, 255).astype(np.uint8)
    final_image = np.transpose(final_image, (1, 2, 0))

    # **Cải thiện hậu xử lý ảnh**
    final_image = exposure.adjust_gamma(final_image, gamma=0.90)  # Điều chỉnh sáng nhẹ
    final_image = exposure.adjust_sigmoid(final_image, cutoff=0.5, gain=3.0)  # Tăng độ tương phản
    final_image = gaussian_filter(final_image, sigma=0.5)  # Giảm noise nhẹ
    final_image = median_filter(final_image, size=1)  # Giảm hạt nhiễu cục bộ
    final_image = uniform_filter(final_image, size=2)  # Làm mượt nhẹ để ảnh tự nhiên hơn

    # Xuất ảnh
    pil_img = Image.fromarray(final_image)
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)
    
    return Response(img_io.getvalue(), media_type="image/png")
