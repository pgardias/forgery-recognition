
"""I trained this model using keras 2.1.1 and tensorflow 1.5.0 w/ GPU
The import statements need to be changed to replace keras with
tensorflow.keras instead.

You do not need to real this CNN as the weights and model are saved already. Therefore you can simply load them in
as shown in the word generation class
"""


from emnist import extract_training_samples,extract_test_samples
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPooling2D
import keras
import numpy as np

trainX, trainY = extract_training_samples('letters')
testX, testY = extract_test_samples('letters')


trainX = trainX.reshape(trainX.shape[0], 28,28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

num_classes = 26

trainY = np.subtract(trainY,1)
testY = np.subtract(testY,1)

trainY = keras.utils.to_categorical(trainY,num_classes)
testY = keras.utils.to_categorical(testY,num_classes)

CNN = Sequential()
CNN.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))

CNN.add(Conv2D(64, (3, 3), activation='relu'))
CNN.add(MaxPooling2D(pool_size=(2, 2)))
CNN.add(Dropout(0.25))

CNN.add(Flatten())
CNN.add(Dense(128, activation='relu'))
CNN.add(Dropout(0.5))

CNN.add(Dense(num_classes, activation='softmax'))

CNN.compile(loss="categorical_crossentropy",optimizer=Adadelta(),metrics="Accuracy")
batch_Size = 128 # change to fit your GPU limit, try as large as possible
epochs = 10 # you can try more epochs but this model seems to cap out around 95% around 8 epochs
log = CNN.fit(trainX,trainY,batch_size=batch_Size,epochs=epochs,verbose=1,validation_data=(testX,testY))

score = CNN.evaluate(trainX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = CNN.to_json()
with open("cnn_eminst.json", "w") as json_file:
    json_file.write(model_json)
CNN.save_weights("weights.h5")
