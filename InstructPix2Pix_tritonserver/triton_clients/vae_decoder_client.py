import torch
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

def decode_vae_latents(latents_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode latents into image using Triton VAE Decoder.
    - latents_tensor shape: (1, 4, 64, 64)
    - returns: image tensor (1, 3, 512, 512)
    """
    latents_np = latents_tensor.cpu().numpy().astype(np.float16)

    inputs = [
        httpclient.InferInput("latents", latents_np.shape, "FP16"),
    ]
    inputs[0].set_data_from_numpy(latents_np)

    outputs = [httpclient.InferRequestedOutput("decoded_image")]

    response = client.infer("instruct_pix2pix_vae_decoder", inputs, outputs=outputs)
    image_np = response.as_numpy("decoded_image")

    return torch.from_numpy(image_np).to("cuda").half()