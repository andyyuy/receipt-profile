import base64
import json

import requests
import torch
from diffusers import StableDiffusionPipeline

OLLAMA_API_URL = "http://localhost:11434/api/generate"
IMG_TO_TXT_MODEL = "moondream"
GENERATE_PROFILE_MODEL = "dolphin-phi"

model_id = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

with open("aldi-receipt.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")


def talk_to_ollama(url, data):

    response = requests.post(
        url, headers={"Content-Type": "application/json"}, data=json.dumps(data)
    )

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        return actual_response
    else:
        return "Error: " + str(response.status_code) + response.text


img_to_txt_data = {
    "model": IMG_TO_TXT_MODEL,
    "prompt": "What are the items on this receipt?",
    "stream": False,
    "images": [encoded_string],
}

response = talk_to_ollama(OLLAMA_API_URL, img_to_txt_data)
if "Error" in response:
    print("Unable to generate image description:", response)
    exit()

print("Receipt contents:", response)

generate_profile_data = {
    "model": GENERATE_PROFILE_MODEL,
    "prompt": f"Here are the purchases: {response}. The imagined sex, race, height, style of dress, age range of such a person is...",
    "stream": False,
}
response = talk_to_ollama(OLLAMA_API_URL, generate_profile_data)
print(response)

response = talk_to_ollama(OLLAMA_API_URL, generate_profile_data)
if "Error" in response:
    print("Unable to generate customer description", response)
    exit()

image = pipe(response + " 4K").images[0]

image.save("output.png")
