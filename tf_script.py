from flask import Flask
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import zscore

app = Flask(__name__)
#get some data
#features, price = load_boston(True)
csv = np.genfromtxt(app.root_path + "\\..\\data\\test_cancer_final.csv", delimiter=",",skip_header=1)
features = csv[:, [32,33,34]]
price = csv[:, [35]]

#standardize features (so all features are on the same scale -- helps for setting penalty):
featuresZ = zscore(features)

#add bias (also known as intercept)
featuresZ_pBias = np.c_[np.ones((featuresZ.shape[0], 1)), featuresZ]


#Pull out a separate test set:
featuresZ_pBias, price = shuffle(featuresZ_pBias, price, random_state=42) #in case order is not random
offset = int(featuresZ_pBias.shape[0] * 0.9)
featuresZ_pBias_test, price_test = featuresZ_pBias[offset:], price[offset:]
featuresZ_pBias, price = featuresZ_pBias[:offset], price[:offset]
#now featuresz_pBias and price contain just 80% of the data


## TensorFlow starts here (above is just setting up data to work with)

import tensorflow as tf
X = tf.constant(featuresZ_pBias, dtype=tf.float32, name="X")
y = tf.constant(price.reshape(-1,1), dtype=tf.float32, name="y")

Xt = tf.transpose(X)
penalty = tf.constant(1.0, dtype=tf.float32, name="penalty")
I = tf.constant(np.identity(featuresZ_pBias.shape[1]), dtype=tf.float32, name="I")

'''beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X) + penalty*I), Xt), y)

### everything above is just definitions of operations to carry out over tensors
### now let's create a session to run the operations
with tf.Session() as sess:
 beta_value = beta.eval()

print(beta_value)'''
print(featuresZ)

#first define learning parameters:
n_epochs = 100
learning_rate = 0.0001

#first define the parameters we are going to find optimal values
beta = tf.Variable(tf.random_uniform([featuresZ_pBias.shape[1], 1], -1., 1.), name = "beta")

#then setup the prediction model's graph:
y_pred = tf.matmul(X, beta, name="predictions")

#Define the cost function for which we will calculate the gradients over beta in order to minimize
penalizedCost = tf.reduce_sum(tf.square(y - y_pred)) + penalty * tf.reduce_sum(tf.square(beta))

#add in gradient calculation
grads = tf.gradients(penalizedCost, [beta])[0]

#specify the operation to take for each epoch of training:
training_op = tf.assign(beta, beta-learning_rate*grads)

#initialize variables (i.e. beta)
init = tf.global_variables_initializer()

### everything above is just definitions of operations to carry out over tensors
### now let's create a session to run the operations
with tf.Session() as sess:
 sess.run(init)
 for epoch in range(n_epochs):
  if epoch %10 == 0: #print debugging output
   print("Epoch", epoch, "; penalizedCost =", penalizedCost.eval())
  sess.run(training_op)
 #done training, get final beta:
 best_beta = beta.eval()

print(best_beta)