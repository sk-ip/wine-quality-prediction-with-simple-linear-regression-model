# importing dependencies.
import numpy as np
import tensorflow as tf

# reading contents from txt file.

data = []
labels = []

file = open("winequalitydataset.txt")
read_file = file.readline().rstrip("\n")

while read_file:
	get_values = read_file.split(";")
	get_values = [float(i) for i in get_values]
	data.append(get_values[0:-1])
	
	label = int(get_values[-1])
	labels.append([float(label)])
	
	read_file = file.readline().rstrip("\n")
    
file.close()

# separating the data into training and testing part.

train_data = data[0:4000]
train_label = labels[0:4000]
test_data = data[4001:]
test_label = labels[4001:]

# defining variables and placeholders.

w = tf.Variable(tf.random_normal([11, 1]))
b = tf.Variable(tf.random_normal([4000, 1]))

x = tf.placeholder(dtype = tf.float32, shape = [None, 11])
y = tf.placeholder(dtype = tf.float32, shape = [None, 1])

# hyperparameters defined.
learning_rate = 0.0000001
noofepochs = 1001

# creating the computation graph.

pred = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# runnung the graph under a session.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for steps in range(1001):
        _,c = sess.run([train, cost], feed_dict = {x:train_data, y:train_label})
        if steps%20==0:
            print('step:',steps,'weights:',sess.run(w),'cost',c)
    print("###################### Optimization Finished ##########################")        
    # testing our model.
    print("####################### Testing ############################")
    pred_test = tf.matmul(test_data, w)
    test_cost = tf.reduce_mean(tf.square(pred_test - test_label))
    error = sess.run(test_cost)
    print('Testing Error(Mean Squared Error):',error)

    

