'''Part 1 - Identify the Frauds with the Self-Organizing Map'''
'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset'''
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

'''Feature Scaling'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

'''Training the SOM'''
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

'''Initialize randomly the weights'''
som.random_weights_init(X)

'''Train the SOM'''
som.train_random(data = X, num_iteration = 100)

'''Visualizing the results / i.e. visualizing the MIDs (mean interneuron distances)'''
from pylab import bone, pcolor, colorbar, plot, show
'''Initialize the window (empty graph panel)'''
bone()

'''Take the transpose of the Mean Interneuron Distances (MID)'''
pcolor(som.distance_map().T)

'''Make a legend for the colors'''
colorbar()

markers = ['o', 's'] 
colors = ['r', 'g'] 
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

'''Get explicit list of likely/potentially fraudulent customers'''

'''Create a dictionary of mappings of winning nodes/customers coords'''
mappings = som.win_map(X)

'''Select the matrix locations for potential fraud from visualization above (all the white squares)'''
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
# frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)

'''Inversed the feature scaling to ge the original numbers back'''
frauds = sc.inverse_transform(frauds)

'''Part 2 - Going from Unsupervised to Supervised Deep Learning'''
'''Creating the matrix of features'''
customers = dataset.iloc[:, 1:].values

'''Creating the dependent variable (i.e. if the customer is fraud or not)'''
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

'''Feature scaling'''
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
customers = standardScaler.fit_transform(customers)

'''Part 3 - Now let's make the ANN!'''
'''Importing the Keras libraries and packages'''
from keras.models import Sequential
from keras.layers import Dense

'''Initializing the ANN'''
classifier = Sequential()

'''First hidden layer'''
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))

'''Adding the output layer'''
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

'''Compiling the ANN'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''Fitting the ANN to the Training set'''
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 100)

'''Predicting the probabilities of frauds'''
y_pred = classifier.predict(customers)

'''one for vertical concatenation'''
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

'''sorts numpy array by that column'''
y_pred = y_pred[y_pred[:, 1].argsort()]

pd.DataFrame(y_pred[-40:], columns = ['Customer Id', 'Probability of Fraud'])