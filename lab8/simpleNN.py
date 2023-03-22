import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.metrics import r2_score

# creating custom data/ x and y values
X = np.arange(-210, 210, 3) 
y = np.arange(-200, 220, 3)+np.random.rand(140)*30

# shape of created data
print(X.shape, y.shape)

plt.figure( figsize = (12,6))
# plotting scattered plot of linear data
plt.scatter(X, y, label = 'Dataset')
plt.legend()
plt.show()

# Splitting training and test data
X_train = X[:110]
y_train = y[:110]
X_test = X[110:]
y_test = y[110:]
# printing the input and output shapes
print(len(X_train), len(X_test))

# size of the plot
plt.figure( figsize = (12,6))
# plotting training set and input data
plt.scatter(X_train, y_train, c='b', label = 'Training data')
plt.scatter(X_test, y_test, c='g', label='Test set')
plt.legend()
plt.show()

# creating model and dense layer
model = tf.keras.Sequential([tf.keras.layers.InputLayer(
    input_shape=1),
    tf.keras.layers.Dense(1)])

number_of_epochs = 300
History = []
r2_scores = []
learning_rates = [0.2, 0.1, 0.01, 0.001]
#learning_rates = [0.1]
for learning in learning_rates:
    #compiling
    model.compile( loss = tf.keras.losses.mae,
                #optimizer = tf.keras.optimizers.SGD(learning_rate=learning),#SGD-> stochastic gradient descent
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning),
                metrics = ['mae'])

    # tensorflow run model/train model on input data
    history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=number_of_epochs)
    History.append(history)

    # Prediction of neural network model
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print('R score is :', r2)
    r2_scores.append(r2)

print(r2_scores)

plt.figure(figsize=(12,6))
# plots training data, test set and predictions
for i in range (len(learning_rates)):
    plt.plot(History[i].history["mae"], label="learning rate = " + str(learning_rates[i]))
plt.legend()
plt.show()

a = float(model.layers[0].get_weights()[0])
b = float(model.layers[0].get_weights()[1])

print("a: " + str(a) + ", b: " + str(b))

y_predict = a*X_train + b

# size of the plot
plt.figure(figsize=(12,6))
# plots training data, test set and predictions
plt.plot(X_train, y_predict, c="r", label="fitted function")
plt.scatter(X_train, y_train, c="b", label="Train data")
plt.scatter(X_test, y_test, c="g", label="Test set")
plt.scatter(X_test, preds, c="r", label="Predictions")
plt.legend()
plt.show()