#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import asyncio
import nest_asyncio
nest_asyncio.apply()

loop = asyncio.get_event_loop()
input_shape = (32, 32)

app = FastAPI()

def load_model():
    model = tf.keras.models.load_model('bc_st_cnn_model.hdf5')
    return model 

model = load_model()


from tensorflow.keras.applications.imagenet_utils import decode_predictions

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def predict(image: Image.Image):
    image = np.asarray(image.resize((32, 32)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 2)[0]
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)
    return response

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, port = 8000, host = '127.0.0.1')
    


# In[ ]:




