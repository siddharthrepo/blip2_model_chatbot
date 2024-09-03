from flask import Flask, request, jsonify, render_template , send_from_directory
from PIL import Image
import os
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datetime import datetime



device = "cuda" if torch.cuda.is_available() else "cpu"


app = Flask(__name__, template_folder='templates')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
).to(device)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

file_uploaded = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_uploaded.append(file.filename)
        print(file_uploaded[-1])
        return jsonify({'message': 'File uploaded successfully', 'path': f'/uploads/{file.filename}'}), 200
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# from blip import processor , model


@app.route('/handle_message', methods=['POST'])
async def handle_message():
    message = request.json['message']
    current_time = datetime.now().time()  # Gets the current time
    print("Current Time:", current_time)
    img_dir = "/home/siddharth/Desktop/SIH/blip_app/uploads/"

    if len(file_uploaded) != 0:
        img_path = os.path.join(img_dir , file_uploaded[-1])
        print(img_path)
    else:
        response = "pls upload a file "
        return jsonify({'response': response})

    img = Image.open(img_path)

    # model batch processing
    # with torch.no_grad():
    #     inputs = message
    #     print(f"prompt taken {inputs}")
    #     batch_inputs = processor(images=img,text=inputs, return_tensors="pt").to(device)
    #     print("Batch inputs processed")
    #     outputs = await asyncio.to_thread(model.generate, **batch_inputs)
    #     print("outputs generated")
    #     results = processor.decode(outputs, skip_special_tokens=True)
    #     print("result generated")

    

    inputs = processor(images=img , return_tensors="pt").to(device, torch.float16)
    print("inputs processed")

# Generate text
    generated_ids = model.generate(**inputs)
    print("id's generated")
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("text generated")

    current_time = datetime.now().time()  # Gets the current time
    print("Current Time:", current_time)
    response = "we'll get back to you shortly"
    return jsonify({'response': generated_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0')