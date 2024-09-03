from transformers import (
    Blip2VisionConfig , 
    Blip2QFormerConfig,
    Blip2Config,
    OPTConfig,
    Blip2ForConditionalGeneration,
)
import matplotlib.pyplot as plt

configuration = Blip2Config()
model = Blip2ForConditionalGeneration(configuration)
configuration = model.config


from transformers import Blip2VisionConfig , Blip2VisionModel

configuration_vision = Blip2VisionConfig()
model = Blip2VisionModel(configuration_vision)

configuration_2 = model.config

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model with float16 precision
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
).to(device)

prompt = "Question: what is the image? Answer:"
inputs = processor(images=image, text = prompt,return_tensors="pt").to(device, torch.float16)

# Generate text
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)


