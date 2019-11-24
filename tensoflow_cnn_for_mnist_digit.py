import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot = True)

batch_size = 128
classes = 10

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_neural_network(x):
    weights = {
	       'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'Out':tf.Variable(tf.random_normal([1024,classes]))
               }
    biases = {
	       'B_conv1':tf.Variable(tf.random_normal([32])),
               'B_conv2':tf.Variable(tf.random_normal([64])),
               'B_fc':tf.Variable(tf.random_normal([1024])),
               'Out':tf.Variable(tf.random_normal([classes]))
               } 

    x = tf.reshape(x,shape=[-1,28,28,1])
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1'])+biases['B_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['B_conv2'])
    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['B_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
              
    output = tf.matmul(fc,weights['Out'])+biases['Out']
    return output
    
def train_nn(x):
    pred = conv_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) )
    optimize = tf.train.AdamOptimizer().minimize(cost)
    main_epoch = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(main_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimize,cost],feed_dict = {x:epoch_x,y:epoch_y})
                epoch_loss+=c
                
            print('Epoch',epoch,'completed out of',main_epoch,'loss',epoch_loss)
        correct = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
        print('Accuracy',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
train_nn(x)
    
    



