from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import nest_asyncio

nest_asyncio.apply()

app = Flask(__name__)

# Load model
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
#pipeline = pipeline.to("cuda")

@app.route('/')
def home():
    return "Welcome to Stable Diffusion API!"

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({"error": "No prompt provided!"}), 400

    # Generate image from text prompt
    image = pipeline(prompt).images[0]

    # Convert image to base64
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
