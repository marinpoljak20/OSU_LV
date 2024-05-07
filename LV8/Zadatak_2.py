import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

model = keras.models.load_model("FCN/")
model.summary()

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_test_s = np.expand_dims(x_test_s, -1)

y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_test_2 = x_test_s.reshape(10000, 784)

y_p = model.predict(x_test_2)

y_test_s_2 = y_test_s.argmax(axis=1)
y_p_2 = y_p.argmax(axis=1)