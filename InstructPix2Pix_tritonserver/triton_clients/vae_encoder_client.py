import torch
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

def encode_vae_latents(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Encode input image tensor into latents using Triton VAE Encoder.
    - image_tensor shape: (1, 3, 512, 512)
    - returns: latents tensor (1, 4, 64, 64)
    """
    image_np = image_tensor.cpu().numpy().astype(np.float16)
    
    inputs = [
        httpclient.InferInput("image", image_np.shape, "FP16"),
    ]
    inputs[0].set_data_from_numpy(image_np)

    outputs = [httpclient.InferRequestedOutput("latents")]

    response = client.infer("instruct_pix2pix_vae_encoder", inputs, outputs=outputs)
    latents_np = response.as_numpy("latents")

    return torch.from_numpy(latents_np).to("cuda").half()