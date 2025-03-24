import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)
#run 1 _ run 2
#prompt = " turn him into cyborg"
prompt = "turn him into a child"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
# images[0]

# Hiển thị ảnh kết quả
images[0].show()

# Hoặc lưu ảnh vào file
images[0].save("D:\\SourceCode\\ProjectOJT\\complete\\OJT_TASK3_LOCAL\\Deploy\\result\\output.jpg")
print("Kết quả đã được lưu vào output.jpg")