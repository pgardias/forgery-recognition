# don't use this file unless you want to mess with the model prediction/evaluation

#similar to letter_CNN.py where you may need to replace keras with tensorflow.keras
from emnist import extract_training_samples,extract_test_samples
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
from numpy.random import randint

trainX, trainY = extract_training_samples('letters')
testX, testY = extract_test_samples('letters')

x = randint(0,12000)
random = trainX[x]

json = open('cnn_emnist.json')
load_model_json = json.read()
json.close()

model = model_from_json(load_model_json)
model.load_weights("emnist_cnn_100.h5")

pred = model.predict(random.reshape(1,28,28,1))
print(np.argmax(pred) + 1)#Add 1 to prediction since we had to shift labels from 1-26 to 0-25 then back for prediction

plt.imshow(random.reshape((28, 28)), cmap = 'binary')
plt.show()
print(trainY[x])

exit(0)