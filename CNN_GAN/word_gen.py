folder = 'BaselineGANletters/Epochs_100/'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

folder = 'Epochs_100/'#folder to pull GANs from

#Generating word dictionary
letters = ['a','b','c','d','e','f','g','h','i','j','k','l',
               'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
letters_dict = {}
for i in range(0,len(letters)):
    letters_dict[letters[i]] = i+1#Storing as 1-26, a=1...

#load CNN model into system
json = open('cnn_emnist.json')
load_model_json = json.read()
json.close()
cnn = tf.keras.models.model_from_json(load_model_json)
cnn.load_weights("emnist_cnn_100.h5")

# same code as the baseline gan model new lines are commented
def make_word_GAN_test(word, example_number, folder, epochs):
    first_letter = True
    for letter in word:
        cap = letter.capitalize()
        model_name = folder + cap + '_generator.h5'
        my_model = tf.keras.models.load_model(model_name, compile=False)
        looksGood = False #bool to determine whether or not the word is legible
        actual_letter = letters_dict.get(letter) #letter value for letter we are trying to create
        noise = tf.random.normal([1, 100])
        generated_image = my_model(noise, training=False)
        while not looksGood:# while the letter doesn't look good
            noise = tf.random.normal([1, 100])
            generated_image = my_model(noise, training=False) #continually develop new letters
            x = generated_image.numpy()#convert to numpy array (what the cnn input is)
            pred = cnn.predict(x)#predict w/ CNN
            predicted_letter = np.argmax(pred) + 1#add 1 to CNN predicition since CNN between 0-25, and letters here 1-26
            if predicted_letter==actual_letter:#if its good change the boolean to break loop
                looksGood = True
        plt.imshow(generated_image[0, :, :, 0], cmap='binary')
        plt.imsave('GANtest.png', generated_image[0, :, :, 0], cmap='binary')
        my_image = Image.open('GANtest.png')

        if first_letter:
            word_array = np.array(my_image)
            first_letter = False

        else:
            word_array = np.hstack((word_array, np.array(my_image)))

    plt.imshow(word_array, cmap='binary')
    filestring = 'EMNIST_Fake' + str(epochs) + 'Epochs/GAN-EMN0' + str(example_number) + word + '.png'
    plt.imsave(filestring, word_array, cmap='binary')

word_list = ['wet', 'pin', 'tab', 'bay',
             'the', 'too', 'red', 'key', 'say',
             'zoo', 'rat', 'run', 'car', 'ten']

num_words = len(word_list)
num_examples = 5

for word in word_list:
    example_count = 0

    while example_count < num_examples:
        #make_word_MNIST_test(word, example_count) we dont' need this as these samples won't change between
        # models since generated from same dataset

        folder = 'Epochs_50/'
        epochs = 50
        make_word_GAN_test(word, example_count, folder, epochs)

        folder = 'Epochs_100/'
        epochs = 100
        make_word_GAN_test(word, example_count, folder, epochs)

        example_count = example_count + 1