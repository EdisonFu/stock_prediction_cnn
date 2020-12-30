
import numpy as np
from PIL import Image

import model as m

filename = "image/predict/1.png"

def predict():
    img = Image.open(filename)

    arr = np.array(img)

    model = m.build((48, 100, 3), 2)
    input = np.array([arr])
    predict = model.predict(input)

    text = np.argmax(predict, axis=1)

    print("the image is", text)
    return {'text': text}
