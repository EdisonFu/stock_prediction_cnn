import os

import numpy as np
from PIL import Image
from six.moves import cPickle

image_folder = 'image/train'
pack_name = 'train_batch'

#empty(shape[], dtype, order]) 依给定的shape, 和数据类型 dtype,  返回一个一维或者多维数组，数组的元素不为空，为随机产生的数据。
data = {'data': np.empty(shape=(0, 48, 100, 3), dtype=float), 'labels': np.empty(shape=(0, 1), dtype=int)}

i = 0
print("start packing", image_folder)
for dirname, dirnames, filenames in os.walk(image_folder):
    for filename in filenames:
        if filename.endswith('.png'):
            im = Image.open(os.path.join(dirname, filename)).convert('RGB')
            arr = np.array(im)   #当使用PIL.Image.open()打开图片后，如果要使用img.shape函数，需要先将image形式转换成array数组
            data['data'] = np.append(data['data'], np.array([arr]), axis=0)

            class_name = os.path.join(dirname).split('/')[-1]
            print("class_name:",class_name)
            class_code = np.array([str(class_name)])
            data['labels'] = np.append(data['labels'], np.array([class_code]), axis=0)
        i += 1
        if i % 1000 == 0:
            print(i, "data formatted")

#print(data['data'].shape) --->  (10000, 48, 100, 3)
#print(data['labels'].shape) ----> (10000, 1)

#cPickle可以对任意一种类型的python对象进行序列化操作，比如list，dict，甚至是一个类的对象等。而所谓的序列化，可理解就是为了能够完整的保存并能够完全可逆的恢复。
#dump： 将python对象序列化保存到本地的文件。  load：载入本地文件，恢复python对象
cPickle.dump(data, open(pack_name, "wb"))
