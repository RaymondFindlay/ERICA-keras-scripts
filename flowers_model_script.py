# Train MobileNet model instance on flowers dataset
# Raymond Findlay 2019

import keras
import numpy as np
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

# import mobilenet model
mobile = keras.applications.mobilenet.MobileNet()

# set paths to image data
train_path = 'flowers/train/'
valid_path = 'flowers/valid/'
test_path = 'flowers/test/'

# prep image data using directory iterators
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), 
                                                         classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 
                                                         batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), 
                                                         classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 
                                                         batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), 
                                                        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], 
                                                        batch_size=10)
														
# modify the default mobilenet
# crop the final five layers from default mobilenet
crop_layers = mobile.layers[-6].output

# append dense output layer with 5 nodes
predicitions = Dense(5, activation='softmax')(crop_layers)

# construct new model with keras functional API with new output nodes
model = Model(inputs=mobile.input, outputs=predicitions)

# retune final five layers of new model
# (all layers before will have weights obtained by training on imagenet)
for layer in model.layers[:-5]:
    layer.trainable = False

# train the new model
# compile the model using the Adam optimiser
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# train model using fit_generator
model.fit_generator(train_batches, steps_per_epoch=45, validation_data=valid_batches, validation_steps=2, epochs=20, verbose=2)

# format labels for test set before predictions
test_labels = test_batches.classes

# make predictions
predictions = model.predict_generator(test_batches, steps=10, verbose=0)

# Confusion matrix setup
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

# plot a confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max() / 2
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

cm_plot_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')