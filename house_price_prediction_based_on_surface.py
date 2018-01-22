import tensorflow as tf 
import numpy as np  #numpy for random number generators & array conversion features
import math #math functions
import matplotlib.pyplot as plt #plot and animate our data
import matplotlib.animation as animation #import animation support

#generate data - house sizes between 1000 and 3500 sq ft
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low = 1000, high=3500, size=num_house)

#generate house price from it's size and some noises

np.random.seed(42)
house_price = house_size * 100.0 +np.random.randint(low=2000,high=7000, size=num_house)


#plot generate house and size
plt.plot(house_size,house_price,'rx') #rx= red x
plt.ylabel('price')
plt.xlabel('size')
plt.show()

#We need to normalize our data 
def normalize(array):
    return (array - array.mean()) / array.std()

#Define number of training samples - 70% in our case
num_train_sample = math.floor(num_house * 0.7)

#define trainning data
train_house_size = np.asarray(house_size[:num_train_sample])
train_price = np.asarray(house_price[:num_train_sample])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#define test data
test_house_size = np.asarray(house_size[num_train_sample:])
test_house_price = np.asarray(house_price[num_train_sample:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)


#Set up the tensorflow placeholders that get updated as we descend down the gradient

tf_house_size = tf.placeholder("float",name="house_size")
tf_price = tf.placeholder("float",name="house_price")

#Define the variables holding the size_factor & price we set during training
#We initialze them with some random values

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offest = tf.Variable(np.random.randn(), name="price_offset")




#Define the operation for the predicting value

tf_price_pred = tf.add(tf.multiply(tf_house_size,tf_size_factor),tf_price_offest)

#Define the loss function (erro) - mean square error

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2*num_train_sample)

#Optimier learning rate - the size of the step down the gradient
learning_rate = 0.1

#Define a gradient descent which will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


#Initialize the Tensorflow's variables
init = tf.global_variables_initializer()

#Launch the graph in the session

with tf.Session() as sess :
    sess.run(init)

    #Set how often to display the number of training iterations & training progress
    display_every = 2
    num_training_iter = 50

    #keep iterating the training data
    for iteration in range(num_training_iter):
        #fit training data
        for (x,y) in zip(train_house_size_norm,train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size:x, tf_price:y})

        if (iteration + 1) % num_training_iter == 0:
            c = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
            "size_factor=", sess.run(tf_size_factor), "prince_offset=", sess.run(tf_price_offest) )


    print("Optimization Finished !!")
    training_cost = sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm, tf_price: train_price_norm})
    print("trained cost=",training_cost, "size_factor=",sess.run(tf_size_factor), "price_offset=",sess.run(tf_price_offest))

    # Plot of training and test data, and learned regression
    
    # get values used to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offest)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()


 