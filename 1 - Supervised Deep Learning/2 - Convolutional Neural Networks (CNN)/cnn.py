
'''Part 1 - Building the CNN'''

'''Importing the Keras libraries and packages'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

'''Initialising the CNN'''
classifier = Sequential()

'''Step 1 - Convolution : Applying several feature detectors'''
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape = (64, 64, 3), activation = 'relu'))

'''Step 2 - Pooling'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''Dropout to avoid overfitting '''
classifier.add(Dropout(0.2))

'''Adding a second convolutional layer'''
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu'))

'''Apply Max Pooling to the 2nd Convolutional Layer'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''Dropout to avoid overfitting '''
classifier.add(Dropout(0.2))

'''Step 3 - Flattening'''
classifier.add(Flatten())

'''Step 4: Create a Full Connection Artificial Neural Network'''
classifier.add(Dense(units = 128, activation = 'relu'))

'''Add a second Full Connection Hidden Layer to increase accuracy and performance results'''
# classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) 
# classifier.add(Dense(units=128, activation='relu', kernel_initializer='uniform')) 

'''Adding output layer'''
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

'''Compiling the CNN'''
"""optimizer=Adam(lr=0.001)"""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''Part 2 - Fitting the CNN (Convolutional Neural Network) to the images'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
# training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory("dataset/test_set", target_size = (64, 64), batch_size = 32, class_mode = 'binary')
# test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')

""" Data augmentation """
"""
from keras.preprocessing.image import ImageDataGenerator
# dataget_train = ImageDataGenerator(rotation_range = 90)
# dataget_train = ImageDataGenerator(vertical_flip=True)
# dataget_train = ImageDataGenerator(height_shift_range=0.5)
dataget_train = ImageDataGenerator(brightness_range=(1,3))
dataget_train.fit(X_train_sample)
"""

classifier.fit_generator(training_set, steps_per_epoch = 100, epochs = 100, validation_data = test_set, validation_steps = 2000, shuffle=True, verbose=1) 
# classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 50, validation_data = test_set, validation_steps = 800/32) 

''' Confusion Matrix '''
"""
from sklearn.metrics import confusion_matrix
import seaborn as sns
predicted_classes = classifier.predict_classes(X_test) 
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)
"""

'''Part 3 - Making new predictions'''
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

