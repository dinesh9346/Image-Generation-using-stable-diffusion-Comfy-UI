import torch
from diffusers import StableDiffusionPipeline

# Load pre-trained model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use "cpu" if no GPU available

# Generate an image
prompt = "an astronaut with a bicycle on moon"
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# Save the image
image.save("astronaut_bicycle.png")
print("Image saved successfully!")
