
# Installing Keras 
#     conda install -c conda-forge keras
#     pip install --upgrade keras
# Installing Theano to exploit parallel computations of processor 
#     pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Install Tensorflow 
#     conda create -n tensorflow python=3.5
#     pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.1-cp35-cp35m-win_amd64.whl --ignore-installed

'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

'''Script Start Time'''
startTime = datetime.now()

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

'''Encoding categorical data (independent variable)'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

'''Using Dummy Encoding'''
"""
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
"""
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)


'''Avoiding the Dummy Variable Trap'''
X = X[:, 1:]

'''
from sklearn.compose import ColumnTransformer

preprocess = ColumnTransformer([
    ("OneHotEncoding", OneHotEncoder(), [1, 2]),
    ("StandardScaler", StandardScaler(), [0, 3, 4, 5, 6, 7, 8, 9])
])
X = preprocess.fit_transform(X)
X = np.delete(X, [0, 3], 1)
'''

'''Splitting the dataset into the Training set and Test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''Part 2 - Now let's make the ANN!'''
'''Import the Keras library and required packages'''
# from ann_classifier.keras.models import Sequential   
# from tensorflow.keras.layers import Dense            
import keras
from keras.models import Sequential
from keras.layers import Dense
'''Initialising the ANN'''
classifier = Sequential()

'''Adding the input layer and the first hidden layer with Dropout'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 
# classifier.add(Dropout(p = 0.1))

'''Adding second hidden layer'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

'''Adding the output layer'''
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.summary()
'''Compiling the ANN'''
"""optimizer=Adam(lr=0.001)"""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


'''Fitting the ANN to the training set'''
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, shuffle=True, verbose=1)

'''Part 3 - Making the predictions, test set results and evaluate the model'''

'''Predicting the Test set results'''
y_pred = classifier.predict(X_test)
print(y_pred.shape)
y_pred = (y_pred > 0.5)


'''Predicting ta single new observation'''
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

'''Making the Confusion Matrix'''
from sklearn.metrics import classification_report, confusion_matrix
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


TN = cm[0][0]; FP = cm[0][1]; FN = cm[1][0]; TP = cm[1][1]
accuracy = (TN + TP) / (TN + FP + FN + TP)
precision = TN / (TN + cm[0][1])
recall = TN / (TN + cm[1][0])
f1_score = 2 * precision * recall / (precision + recall)
print("accuracy:", accuracy, "\nprecision:", precision, "\nrecall:", recall, "\nf1_score:", f1_score)
'''Part 4 - Evaluating, Improving and Tuning the ANN'''
