from __future__ import print_function

import os

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential


def build(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    # keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
    # 参数：
    # lr: float >= 0. 学习率 Learning rate
    # momentum: float >= 0. 参数更新动量 parameter updates momentum
    # decay: float >= 0. 学习率每次更新的下降率 Learning rate decay over each update
    # nesterov: boolean.是否应用 Nesterov 动量 whether to apply Nesterov momentum
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    #categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if os.path.isfile('weights.hdf5'):
        model.load_weights('weights.hdf5')

    return model


#學習資料：https://blog.csdn.net/u011746554/article/details/74393922