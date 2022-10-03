import json, os
from io import BytesIO
import base64
from dalle_model import DalleModel

def init(): 
  global dalle_model
  dalle_model = DalleModel('Mega_full')
  dalle_model.generate_images("CN tower is flying to Mars", 1)

def run(data): 
  print(data)
  input_data = json.loads(data)
  # print(input_data)
  text_prompt = input_data["text"]
  num_images = input_data["num_images"]
  generated_imgs = dalle_model.generate_images(text_prompt, num_images)

  returned_generated_images = []
  
  for idx, img in enumerate(generated_imgs):
      buffered = BytesIO()
      img.save(buffered, format='jpeg')
      img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
      returned_generated_images.append(img_str)

  print(f"Created {num_images} images from text prompt [{text_prompt}]")
  
  response = {'generatedImgs': returned_generated_images}
  return json.dumps(response)
