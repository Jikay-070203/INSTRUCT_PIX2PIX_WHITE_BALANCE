import numpy as np
import torch
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

def run_unet(latents, timestep, encoder_hidden_states):
    inputs = [
        httpclient.InferInput("latent_model_input", latents.shape, "FP16"),
        httpclient.InferInput("timestep", (latents.shape[0],), "FP16"),
        httpclient.InferInput("encoder_hidden_states", encoder_hidden_states.shape, "FP16"),
    ]
    inputs[0].set_data_from_numpy(latents.cpu().numpy())
    inputs[1].set_data_from_numpy(np.full((latents.shape[0],), timestep.item(), dtype=np.float16))
    inputs[2].set_data_from_numpy(encoder_hidden_states.detach().cpu().numpy())
    
    outputs = [httpclient.InferRequestedOutput("noise_pred")]

    response = client.infer("instruct_pix2pix_unet", inputs, outputs=outputs)
    noise_pred = response.as_numpy("noise_pred")
    return torch.from_numpy(noise_pred).to("cuda").half()