
import numpy as np
from PIL import Image

import model as m

filename = "image/predict/predict.png"

def predict():
    img = Image.open(filename).convert('RGB')

    arr = np.array(img)

    model = m.build((48, 100, 3), 2)
    print("1")
    input = np.array([arr])
    print("input:",input)
    predict = model.predict(input)
    print("2")
    text = np.argmax(predict, axis=1)

    print("the image is", text)
    return {'text': text}

predict()