from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "A futuristic cityscape at night with flying cars."
image = pipe(prompt).images[0]
plt.imshow(image)
plt.axis('off')
plt.show()