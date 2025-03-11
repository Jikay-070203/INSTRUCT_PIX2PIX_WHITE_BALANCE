from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer

app = FastAPI()
triton_client = httpclient.InferenceServerClient("localhost:8000")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

@app.post("/predict/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    """Nhận ảnh và prompt, xử lý với Triton Server"""
    
    # 1️⃣ Mã hóa văn bản với Text Encoder
    tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="np")["input_ids"].astype(np.int64)
    text_input = httpclient.InferInput("input_text", tokens.shape, "INT64")  # Đổi "input_ids" → "input_text"
    text_input.set_data_from_numpy(tokens)

    text_response = triton_client.infer("text_encoder", inputs=[text_input],
                                        outputs=[httpclient.InferRequestedOutput("text_embeddings")])
    text_embedding = text_response.as_numpy("text_embeddings").astype(np.float32)  # Đổi FP16 → FP32

    # 2️⃣ Mã hóa ảnh với VAE Encoder
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0  # Đổi FP16 → FP32
    image_np = np.transpose(image_np, (2, 0, 1))[None, :, :, :]

    vae_input = httpclient.InferInput("image", image_np.shape, "FP32")  # Đổi FP16 → FP32
    vae_input.set_data_from_numpy(image_np)

    vae_response = triton_client.infer("vae_encoder", inputs=[vae_input],
                                       outputs=[httpclient.InferRequestedOutput("latent")])  # Đổi "latents" → "latent"
    latent = vae_response.as_numpy("latent").astype(np.float32)  # Đổi FP16 → FP32

    # 3️⃣ Gửi dữ liệu vào UNet để xử lý nhiễu
    timestep = np.array([1], dtype=np.float16)  # UNet yêu cầu FP16

    latent_input = httpclient.InferInput("latents", latent.shape, "FP16")
    latent_input.set_data_from_numpy(latent.astype(np.float16))  # Chuyển đổi sang FP16

    timestep_input = httpclient.InferInput("timestep", timestep.shape, "FP16")  # Đổi "time_steps" → "timestep"
    timestep_input.set_data_from_numpy(timestep)

    text_emb_input = httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16")  # Đổi FP32 → FP16
    text_emb_input.set_data_from_numpy(text_embedding.astype(np.float16))

    unet_response = triton_client.infer("unet",
                                        inputs=[latent_input, timestep_input, text_emb_input],
                                        outputs=[httpclient.InferRequestedOutput("predicted_noise")])
    denoised_latents = unet_response.as_numpy("predicted_noise").astype(np.float32)  # Đổi FP16 → FP32

    # 4️⃣ Decode latent space thành ảnh
    vae_dec_input = httpclient.InferInput("latent", denoised_latents.shape, "FP32")  # Đổi "latents" → "latent"
    vae_dec_input.set_data_from_numpy(denoised_latents)

    vae_dec_response = triton_client.infer("vae_decoder",
                                           inputs=[vae_dec_input],
                                           outputs=[httpclient.InferRequestedOutput("image")])  # Đổi "decoded_image" → "image"
    generated_image = vae_dec_response.as_numpy("image")[0]

    generated_image = np.clip(np.transpose(generated_image, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

    # 5️⃣ Trả về ảnh kết quả
    pil_img = Image.fromarray(generated_image)
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(img_io.getvalue(), media_type="image/png")
