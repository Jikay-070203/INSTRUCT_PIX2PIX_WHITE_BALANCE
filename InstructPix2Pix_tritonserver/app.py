import os
import torch
from datetime import datetime
from fastapi import FastAPI, UploadFile
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse
from triton_clients.vae_encoder_client import encode_vae_latents
from triton_clients.unet_client import run_unet
from triton_clients.vae_decoder_client import decode_vae_latents

app = FastAPI()

model_id = "hoanguyenthanh07/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.eval().to("cuda")
image_processor = pipe.image_processor
scheduler = pipe.scheduler
vae_scale_factor = pipe.vae_scale_factor

@app.post("/inference")
async def infer(prompt: str, image: UploadFile, negative_prompt: str = ""):
    # Create folders
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("intermediate", exist_ok=True)  # For optional debug output

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    input_filename = f"input/input_{timestamp}.jpeg"
    output_filename = f"output/output_{timestamp}.jpeg"

    # Read and preprocess input image
    image_bytes = await image.read()
    init_image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    init_image.save(input_filename, format="JPEG")

    image_tensor = image_processor.preprocess(init_image).to("cuda").half()

    # Step 1: Encode image to latents via Triton
    image_latents = encode_vae_latents(image_tensor)

    # Step 2: Encode prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_embeds = text_encoder(text_inputs.input_ids)[0]

    # Classifier-free guidance
    do_classifier_free_guidance = True
    if do_classifier_free_guidance:
        neg_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        negative_prompt_embeds = text_encoder(neg_inputs.input_ids)[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds, negative_prompt_embeds])

    prompt_embeds = prompt_embeds.repeat(1, 1, 1)

    # Step 3: Sampling
    scheduler.set_timesteps(10, device="cuda")
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16) * scheduler.init_noise_sigma
    image_latents = torch.cat([image_latents] * 3, dim=0)

    guidance_scale = 7.5
    image_guidance_scale = 1.5

    for i, t in enumerate(scheduler.timesteps):
        print(f"[{i+1}/{len(scheduler.timesteps)}] Timestep: {t}")

        latent_model_input = latents.repeat(3, 1, 1, 1)
        scaled_latent = scheduler.scale_model_input(latent_model_input, t)
        scaled_latent = torch.cat([scaled_latent, image_latents], dim=1)

        noise_pred = run_unet(scaled_latent, t, prompt_embeds)
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)

        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        #Optional: Save intermediate image
        decoded_step_image = decode_vae_latents(latents)
        intermediate_pil = image_processor.postprocess(decoded_step_image, output_type="pil")[0]
        intermediate_pil.save(f"intermediate/step_{i+1:02d}.jpeg", format="JPEG")

        #Optional for low VRAM systems:
        torch.cuda.empty_cache()

    # Step 4: Decode and return final image
    decoded_image = decode_vae_latents(latents)
    final_image = image_processor.postprocess(decoded_image, output_type="pil")[0]

    final_image.save(output_filename, format="JPEG")

    buf = BytesIO()
    final_image.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")
