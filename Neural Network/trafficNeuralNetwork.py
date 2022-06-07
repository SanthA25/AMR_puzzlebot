import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#####################################################################
# Pre-processing
#####################################################################

# Load images
images = []     # Images
labels = []     # Corresponding labels

rootpath = "Data2/MyData/"

# Loop over all 43 classes
for c in range(0,43):
    prefix = rootpath + format(c, '05d') + '/'

    # Load annotations file
    gtFile = open(prefix + 'GT-'+ format(c,'05d') + '.csv', 'r')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)

    # Loop over all images in current annotations file  the 1th column is the filename, the 8th column is the label)
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(row[7])
    gtFile.close()

labels = [int(i) for i in labels]

# Process images
processed_images = []

for im in images:
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im_smoothed = cv2.blur(im_gray,(2,2))
    im_scaled = cv2.resize(im_smoothed, (32, 32), interpolation = cv2.INTER_AREA)
    processed_images.append(im_scaled)

################################
# Data preparation for CNN
################################

num_classes = 43

x=np.array(processed_images)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
x = x/255.


y=np.array(labels,dtype=int)
yc = np_utils.to_categorical(y)
input_shape = (x.shape[1], x.shape[2], 1)
# print('\n',input_shape,'\n')

################################
# Model Evaluation
################################

# Model definition

model = Sequential()

model.add(Conv2D(60, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
model.add(Conv2D(60, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(30, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(30, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(500, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='Adam')

# Separate training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, stratify=y)

yc_train = np_utils.to_categorical(y_train)
yc_test = np_utils.to_categorical(y_test)

# Train model
model.fit(x_train, yc_train, epochs=10, batch_size=128, verbose=1, validation_split=0.2)

# Evaluate model
y_pred = np.argmax(model.predict(x_test), axis=-1)

target_names = [str(i) for i in range(43)]
print(classification_report(y_test, y_pred, target_names=target_names))

################################################################
# Final model (use all the available data for training)
################################################################

# Model definition

model = Sequential()

model.add(Conv2D(60, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
model.add(Conv2D(60, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(30, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(30, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(500, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='Adam')

# Train Model
model.fit(x, yc, epochs=1, batch_size=128, verbose=1, validation_split=0.1)

# Save model
model.save('GTSRB_cnn.h5')
