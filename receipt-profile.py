import base64
import json

import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
IMG_TO_TXT_MODEL = "moondream"
GENERATE_PROFILE_MODEL = "dolphin-phi"

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

generate_profile_data = {
    "model": GENERATE_PROFILE_MODEL,
    "prompt": f"Using this list of receipt items, can you generate an imagined list of physical characteristics of this customer? State their age-range, sex, style of dress, race, height, body modification (if any), hair style, etc. Keep the list brief and concise. Here is the list of items: {response}",
    "stream": False,
}

response = talk_to_ollama(OLLAMA_API_URL, generate_profile_data)
if "Error" in response:
    print("Unable to generate customer description", response)
    exit()

payload = {"prompt": response, "steps": 5}
response = requests.post(url="http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
r = response.json()
print(r)

if "images" in r:
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(r["images"][0]))
