folder = 'BaselineGANletters/Epochs_100/'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from random import randint
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


def filtered_images(labels_array, images_array, letter):
    if (letter == 'A' or letter == 'a'):
        number = 1
    if (letter == 'B' or letter == 'b'):
        number = 2
    if (letter == 'C' or letter == 'c'):
        number = 3
    if (letter == 'D' or letter == 'd'):
        number = 4
    if (letter == 'E' or letter == 'e'):
        number = 5
    if (letter == 'F' or letter == 'f'):
        number = 6
    if (letter == 'G' or letter == 'g'):
        number = 7
    if (letter == 'H' or letter == 'h'):
        number = 8
    if (letter == 'I' or letter == 'i'):
        number = 9
    if (letter == 'J' or letter == 'j'):
        number = 10
    if (letter == 'K' or letter == 'k'):
        number = 11
    if (letter == 'L' or letter == 'l'):
        number = 12
    if (letter == 'M' or letter == 'm'):
        number = 13
    if (letter == 'N' or letter == 'n'):
        number = 14
    if (letter == 'O' or letter == 'o'):
        number = 15
    if (letter == 'P' or letter == 'p'):
        number = 16
    if (letter == 'Q' or letter == 'q'):
        number = 17
    if (letter == 'R' or letter == 'r'):
        number = 18
    if (letter == 'S' or letter == 's'):
        number = 19
    if (letter == 'T' or letter == 't'):
        number = 20
    if (letter == 'U' or letter == 'u'):
        number = 21
    if (letter == 'V' or letter == 'v'):
        number = 22
    if (letter == 'W' or letter == 'w'):
        number = 23
    if (letter == 'X' or letter == 'x'):
        number = 24
    if (letter == 'Y' or letter == 'y'):
        number = 25
    if (letter == 'Z' or letter == 'z'):
        number = 26

    label_filter = np.where(labels_array == number)
    labelled_images = images_array[label_filter]

    return labelled_images


def make_word_MNIST_test(word, example_number):
    from emnist import extract_training_samples
    images_array, labels_array = extract_training_samples('letters')

    first_letter = True

    for letter in word:

        if first_letter:
            letter_set = filtered_images(labels_array, images_array, str(letter))
            num_letters = letter_set.shape[0]
            this_letter = randint(0, num_letters)
            # print(num_letters)
            plt.imshow(letter_set[this_letter].reshape((28, 28)), cmap='binary')

            word_array = letter_set[this_letter].reshape((28, 28))
            first_letter = False

        else:
            letter_set = filtered_images(labels_array, images_array, str(letter))
            num_letters = letter_set.shape[0]
            this_letter = randint(0, num_letters)
            plt.imshow(letter_set[this_letter].reshape((28, 28)), cmap='binary')
            word_array = np.hstack((word_array, letter_set[this_letter].reshape((28, 28))))

    filestring = 'real_words/GAN-EMN0' + str(example_number) + word + '.png'
    plt.imshow(word_array, cmap='binary')
    plt.imsave(filestring, word_array, cmap='binary')

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

    #plt.imshow(word_array, cmap='binary')
    filestring = 'EMNIST_Fake' + str(epochs) + 'Epochs/GAN-EMN0' + str(example_number) + word + '.png'
    plt.imsave(filestring, word_array, cmap='binary')



def make_sentence_GAN(sentence,example_num,folder,epochs):
    words = sentence.split()
    files = []
    for w in words:
        make_word_GAN_test(w,example_num,folder,epochs)
        files.append('EMNIST_Fake' + str(epochs) + 'Epochs/GAN-EMN0' + str(example_num) + w + '.png')

    img = cv2.imread(files[0])
    padding = 255 * np.ones((28, 28, 3), dtype='uint8')
    img = cv2.hconcat([img, padding])
    for f in range(1,len(files)):
        temp = cv2.imread(files[f])
        padding = 255 * np.ones((28, 28, 3), dtype='uint8')
        img = cv2.hconcat([img, padding,temp])
    cv2.imwrite(sentence + "_" + str(example_num),img)

#make_sentence_GAN("i like turtles",1,'Epochs_100/',100)
#make_word_GAN_test("test",1,'Epochs_100/',100)