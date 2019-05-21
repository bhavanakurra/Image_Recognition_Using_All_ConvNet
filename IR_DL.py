
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os


# In[2]:


x = tf.placeholder(shape = [None, 32, 32, 3], dtype = tf.float32)
y = tf.placeholder(shape = [None, 10], dtype = tf.float32)


# In[3]:


conv1_weights = tf.get_variable(name="conv1_weights", shape=[3, 3, 3, 96], dtype=tf.float32, 
                               initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv1_biases = tf.get_variable(name="conv1_biases", shape=[96], dtype=tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed=None, dtype = tf.float32))

conv1 = tf.nn.conv2d( x, conv1_weights, strides = [1, 1, 1, 1], padding='SAME')
conv1_out = tf.nn.relu(conv1 + conv1_biases)
print(conv1_out)


# In[4]:


conv2_weights = tf.get_variable( name = "conv2_weights", shape=[3, 3, 96, 96], dtype=tf.float32,
                               initializer = tf.contrib.layers.xavier_initializer(uniform= True, seed=None, dtype=tf.float32))

conv2_biases = tf.get_variable( name = "conv2_biases", shape=[96], dtype=tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv2 = tf.nn.conv2d( conv1_out, conv2_weights, strides = [1, 1, 1, 1], padding='SAME')
conv2_out = tf.nn.relu(conv2 + conv2_biases)
print(conv2_out)


# In[5]:


conv3_pool_w = tf.get_variable( name="conv3_pool_w", shape=[3, 3, 96, 96], dtype = tf.float32,
                            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv3_pool_b =tf.get_variable( name = "conv3_pool_b", shape=[96], dtype=tf.float32,
                             initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype= tf.float32))

conv3_pool = tf.nn.conv2d( conv2_out, conv3_pool_w, strides=[1, 2, 2, 1], padding="SAME")
conv3_out = tf.nn.relu( conv3_pool + conv3_pool_b)
print(conv3_out)


# In[6]:


conv4_weights = tf.get_variable( name = "conv4_weights", shape=[3, 3, 96, 192], dtype= tf.float32,
                               initializer= tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv4_biases = tf.get_variable( name = "conv4_biases", shape=[192], dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv4 = tf.nn.conv2d( conv3_out, conv4_weights, strides = [1, 1, 1, 1], padding="SAME")
conv4_out = tf.nn.relu(conv4 + conv4_biases)
print(conv4_out)


# In[7]:


conv5_weights = tf.get_variable( name = "conv5_weights", shape = [3, 3, 192, 192], dtype=tf.float32,
                               initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed=None, dtype=tf.float32))
conv5_biases = tf.get_variable( name = "conv5_biases", shape = [192], dtype=tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype= tf.float32))

conv5 = tf.nn.conv2d( conv4_out, conv5_weights, strides=[1, 1, 1, 1 ], padding='SAME')
conv5_out = tf.nn.relu( conv5 + conv5_biases)
print(conv5_out)


# In[8]:


conv6_pool_w = tf.get_variable( name = "conv6_pool_w", shape = [3, 3, 192, 192], dtype=tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv6_pool_b = tf.get_variable( name = "conv6_pool_b", shape=[192], dtype=tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv6_pool = tf.nn.conv2d( conv5_out, conv6_pool_w, strides=[1, 2, 2, 1], padding="SAME")
conv6_out = tf.nn.relu( conv6_pool + conv6_pool_b)
print(conv6_out)


# In[9]:


conv7_weights = tf.get_variable( name="conv7_weights", shape = [3, 3, 192, 192], dtype=tf.float32,
                               initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv7_biases = tf.get_variable( name="conv7_biases", shape = [192], dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype= tf.float32))

conv7 = tf.nn.conv2d( conv6_out, conv7_weights, strides = [1, 1, 1, 1], padding="VALID")
conv7_out = tf.nn.relu( conv7 + conv7_biases)
print(conv7_out)


# In[10]:


conv8_fc_weights = tf.get_variable( name="conv8_fc_weights", shape= [1, 1, 192, 192], dtype=tf.float32,
                                  initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv8_fc_biases = tf.get_variable( name = "conv8_fc_biases", shape = [192], dtype=tf.float32,
                                 initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv8_fc = tf.nn.conv2d( conv7_out, conv8_fc_weights, strides = [1, 1, 1, 1], padding="SAME")
conv8_out = tf.nn.relu( conv8_fc + conv8_fc_biases)
print(conv8_out)


# In[11]:


conv9_fc_weights = tf.get_variable( name = "conv9_fc_weights", shape = [1, 1, 192, 10], dtype = tf.float32,
                                  initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv9_fc_biases = tf.get_variable(name = "conv9_fc_biases", shape=[10], dtype=tf.float32,
                                  initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

conv9_fc = tf.nn.conv2d( conv8_out, conv9_fc_weights, strides=[1, 1, 1, 1], padding="SAME")
conv9_out = tf.nn.relu(conv9_fc + conv9_fc_biases)
print(conv9_out)


# In[12]:


avg = tf.nn.avg_pool(conv9_out, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding="VALID" )
print(avg)


# In[13]:


avg_reshape = tf.reshape(avg, [-1, 10])
output = tf.nn.softmax(avg_reshape)

print(avg_reshape)
print(output)


# In[14]:


loss = tf.nn.softmax_cross_entropy_with_logits_v2( labels=y, logits=avg_reshape)
optimizer = tf.train.AdamOptimizer(0.0001)
train_step = optimizer.minimize(loss)

correct_predictions = tf.equal( tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean( tf.cast( correct_predictions, tf.float32 ))


# In[15]:


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

save_path = ""
data_path = "D:\\deep\\cifar-10-batches-py"


# In[16]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data'] 
        labels = dict[b'labels']
        
        images_data = []
        images_label = []
        
        for i in range(len(images)):
            raw = images[i]
            ch1 = np.reshape(raw[:1024], [32, 32, 1])
            ch2 = np.reshape(raw[1024: 2048], [32, 32, 1])
            ch3 = np.reshape(raw[2048: 3072], [32, 32, 1])
            
            temp1 = np.append(ch1, ch2, axis=2)
            temp2 = np.append(temp1, ch3, axis=2)
            
            tem_labels = np.zeros(shape=[10])
            tem_labels[labels[i]] = 1
            
            images_data.append(temp2)
            images_label.append(tem_labels)
            
        images_data = np.array(images_data)
        images_label = np.array(images_label)
    return images_data, images_label


# In[17]:


train_images = []
train_labels = []

for j in range(1,6):
    file_name = "data_batch_" + str(j)
    images, labels = unpickle(os.path.join(data_path, file_name))
    train_images.append(images)
    train_labels.append(labels)
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(train_images.shape)
print(train_labels.shape)

test_images, test_labels = unpickle(os.path.join(data_path, "test_batch"))
print(test_images.shape)


# In[ ]:


sess.run(init)
for i in range(100):
    print("iteration " + str(i) )

    if(i%10 == 0):
        test_accuracy = 0
        
        for k in range(10):
            test_accuracy = test_accuracy + accuracy.eval(feed_dict = { x: test_images[k*1000 : (k+1)*1000], y:test_labels[k*1000 : (k+1)*1000]})
        
        print("test_accuray     "+ str(test_accuracy/10))
    
    for j in range(5):
        
        for k in range(10):
            
            #train_accuracy1 = accuracy.eval(feed_dict = { x: train_images[j][k*1000 : (k+1)*1000], y:train_labels[j][k*1000 : (k+1)*1000]})

            train_step.run( feed_dict = { x: train_images[j][k*1000 : (k+1)*1000], y: train_labels[j][k*1000 : (k+1)*1000]})

            #train_accuracy = accuracy.eval(feed_dict = { x: train_images[j][k*1000 : (k+1)*1000], y:train_labels[j][k*1000 : (k+1)*1000]})
            #print("             batch "+ str((j)*10000 + k*1000)  + " before "+ str(train_accuracy1) + " after "+ str(train_accuracy))
            print("             batch "+ str((j)*10000 + k*1000))

    

