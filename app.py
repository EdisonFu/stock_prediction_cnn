
import numpy as np
from PIL import Image

import model as m

filename = "image/predict/predict.png"

def predict():
    img = Image.open(filename).convert('RGB')

    arr = np.array(img)

    model = m.build((224, 224, 3), 2)
    input = np.array([arr])
    predict = model.predict(input)
    print("predict:",predict)
    text = np.argmax(predict, axis=1)

    print("the image is", text)
    return {'text': text}

predict()