# Brian Kilburn
# learning tensorflow
# source: https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
#from __future__ import print_function
#hello = tf.constant('Hello, Tensorflow!')
#sess = tf.Session()
#print(sess.run(hello))

#here you will not see the computed values it will print just the constant(value never changes) with no input
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly
print(node1, node2)

#to show input run the computation in a session
sess = tf.Session()
print(sess.run([node1, node2]))

#combine tensor nodes to produce a new graph
node3 = tf.add(node1, node2)
print('node3: ', node3)
print("sess.run(node3):", sess.run(node3))

#external input placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)

#feed values to the placeholders
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

#add another complex operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

#to make a model trainable it must take arbitrary input to train,
#modify to get new output with the same input. Variables allow us to add
#trainable parameters to a graph. these are not constant they can change.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

#this operation will initialize all variables in a tensorflow program, this operation
#is a handle to the tensorflow sub-graph that initializes all global variables, they remain 
#uninitialized until sess.run(init) is called
init = tf.global_variables_initializer()
sess.run(init)

#here we can evaluate several values for x since it is a placeholder
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#measure loss from linear_model to provided data
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#re-assign variables
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0 ,-1, -2, -3]}))
