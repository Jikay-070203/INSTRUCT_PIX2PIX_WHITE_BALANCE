from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer
import logging

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
    tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="np")["input_ids"].astype(np.int64)
    text_input = httpclient.InferInput("input_text", tokens.shape, "INT64")
    text_input.set_data_from_numpy(tokens)

    text_response = triton_client.infer("text_encoder", inputs=[text_input],
                                        outputs=[httpclient.InferRequestedOutput("text_embeddings")])
    text_embedding = text_response.as_numpy("text_embeddings").astype(np.float32)
    
    # 1Kiểm tra text_embedding
    logger.info(f"Bước thử 1: Text embedding shape: {text_embedding.shape}, min: {text_embedding.min()}, max: {text_embedding.max()}")

    #2Mã hóa ảnh với VAE Encoder
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512), Image.LANCZOS)  #  Sử dụng bộ lọc tốt hơn

    
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))[None, :, :, :]

    vae_input = httpclient.InferInput("image", image_np.shape, "FP32")
    vae_input.set_data_from_numpy(image_np)

    vae_response = triton_client.infer("vae_encoder", inputs=[vae_input],
                                       outputs=[httpclient.InferRequestedOutput("latent")])
    latent = vae_response.as_numpy("latent").astype(np.float32)
    
    # 3 Kiểm tra latent
    logger.info(f"Bước thử 3: Latent shape: {latent.shape}, min: {latent.min()}, max: {latent.max()}")

    
    # 3.Gửi dữ liệu vào UNet để xử lý nhiễu
    timestep = np.array([1], dtype=np.float16)
    
    latent_input = httpclient.InferInput("latents", latent.shape, "FP16")
    latent_input.set_data_from_numpy(latent.astype(np.float16))
    
    timestep_input = httpclient.InferInput("timestep", timestep.shape, "FP16")
    timestep_input.set_data_from_numpy(timestep)
    
    text_emb_input = httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16")
    text_emb_input.set_data_from_numpy(text_embedding.astype(np.float16))
    
    unet_response = triton_client.infer("unet",
                                        inputs=[latent_input, timestep_input, text_emb_input],
                                        outputs=[httpclient.InferRequestedOutput("predicted_noise")])
    denoised_latents = unet_response.as_numpy("predicted_noise").astype(np.float32)

    
    # 4Bước thử 4: Kiểm tra denoised_latents
    logger.info(f"Bước thử 4: Denoised latents shape: {denoised_latents.shape}, min: {denoised_latents.min()}, max: {denoised_latents.max()}")

    # 44Decode latent space thành ảnh
    vae_dec_input = httpclient.InferInput("latent", denoised_latents.shape, "FP32")
    vae_dec_input.set_data_from_numpy(denoised_latents)

    vae_dec_response = triton_client.infer("vae_decoder",
                                           inputs=[vae_dec_input],
                                           outputs=[httpclient.InferRequestedOutput("image")])
    generated_image = vae_dec_response.as_numpy("image")[0]
    
    # 5Kiểm tra generated_image
    logger.info(f"Bước thử 5: Generated image shape: {generated_image.shape}, min: {generated_image.min()}, max: {generated_image.max()}")

    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())  # Chuẩn hóa về [0,1]
    generated_image = np.clip(np.transpose(generated_image, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

    # 55Trả về ảnh kết quả
    pil_img = Image.fromarray(generated_image)
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(img_io.getvalue(), media_type="image/png")