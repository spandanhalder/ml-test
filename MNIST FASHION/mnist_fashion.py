# Importing Tensorflow
import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
# Only Showing Error
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# Import Normalize
from additionalfunctions import normalize
# Import Math
import math
import numpy as np
#import matplotlib.pyplot as plt

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# Printing Number of Datasets
num_train_dataset = len(train_dataset)
num_test_dataset = len(test_dataset)
print("Number of Train Dataset = ",num_train_dataset)
print("Number of Test Dataset = ",num_test_dataset)

# The map function applies the normalize function to each element in the train
# and test datasets (Converting Value of (0,255) -> (0,1))
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

model = tf.keras.Sequential([
    # Output Nodes [Hidden]= 32, Kernel = 3x3 , Padding = Same as Original Image
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,input_shape=(28, 28, 1)),
    # Pool Sizze = 2x2, Stride = 2, Selecting Pixel with Greatest Value [Reduce Size]
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    # Output Nodes [Hidden]= 64, Kernel = 3x3 , Padding = Same as Original Image
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    # Pool Sizze = 2x2, Stride = 2, Selecting Pixel with Greatest Value [Reduce Size]
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    # Takes Previous Data and Converts to 1D array. [28x28 pixel = 784 pixel]
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # 10 Possible Output Classes, Softmax for Probabilistic Distribution
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# 32 Examples at a time
BATCH_SIZE = 32
# Suffle the Order, So that model can't learn from order.
train_dataset = train_dataset.cache().repeat().shuffle(num_train_dataset).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# Epochs = 10
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_dataset/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_dataset/32))
print('Accuracy On Test Dataset = ', test_accuracy)

# Taking First Batch Containg 32 Images
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

#print(predictions.shape)

for i in range(BATCH_SIZE):
    #print(predictions[i])
    predictedValue = np.argmax(predictions[i])
    actualValue = test_labels[i]
    print("Predicted = {} | Original = {}".format(class_names[predictedValue],class_names[actualValue]))
    if actualValue == predictedValue:
        print("MATCHED")
    else:
        print("NOT MATCHED")
