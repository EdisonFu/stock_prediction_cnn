
import numpy as np
from PIL import Image

import model as m

filename = ""

def predict():
    img = Image.open(filename)

    arr = np.array(img)

    model = m.build((36, 30, 3), 2)
    input = np.array([arr])
    predict = model.predict(input)

    text = np.argmax(predict, axis=1)

    print("the image is", text)
    return {'text': text}
