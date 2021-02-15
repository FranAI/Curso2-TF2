# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:51:34 2021

@author: Usuario
"""

#### PACKAGE IMPORTS ####

# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten, Add
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# If you would like to make further imports from tensorflow, add them here





# Load and preprocess the Fashion-MNIST dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_images = train_images[:5000] / 255.
train_labels = train_labels[:5000]

test_images = test_images / 255.

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]





# Create Dataset objects for the training and test sets

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)






# Get dataset labels

image_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']






#### GRADED CELL ####

# Complete the following class. 
# Make sure to not change the class or method names or arguments.

class ResidualBlock(Layer):

    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure 
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.batch_norm_1 = BatchNormalization(input_shape = input_shape)
        self.conv_2D_1 = Conv2D(filters = input_shape[-1], kernel_size = (3,3), padding = 'SAME')
        self.batch_norm_2 = BatchNormalization()
        self.conv_2D_2 = Conv2D(filters = input_shape[-1], kernel_size = (3,3), padding = 'SAME')       
        
        
    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        h = self.batch_norm_1(inputs, training = training)
        h = tf.nn.relu(h)
        h = self.conv_2D_1(h)
        h = self.batch_norm_2(h, training = training)
        h = tf.nn.relu(h)
        h = self.conv_2D_2(h)
        output = tf.add(inputs, h)
        return output
        
    
    
    
# Test your custom layer - the following should create a model using your layer

test_model = tf.keras.Sequential([ResidualBlock(input_shape=(28, 28, 1), name="residual_block")])
test_model.summary()





#### GRADED CELL ####

# Complete the following class. 
# Make sure to not change the class or method names or arguments.

class FiltersChangeResidualBlock(Layer):

    def __init__(self, out_filters, **kwargs):
        """
        The class initialiser should call the base class initialiser, passing any keyword
        arguments along. It should also set the number of filters as a class attribute.
        """
        super(FiltersChangeResidualBlock, self).__init__(**kwargs) 
        self.out_filters = out_filters
        
        
        
    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure 
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.batch_norm_1 = BatchNormalization(input_shape = input_shape)
        self.conv_2D_1 = Conv2D(filters = input_shape[-1], kernel_size = (3,3), padding = 'SAME', activation = None)
        self.batch_norm_2 = BatchNormalization()
        self.conv_2D_2 = Conv2D(filters = self.out_filters, kernel_size = (3,3), padding = 'SAME', activation = None)          
        self.conv_2D_3 = Conv2D(filters = self.out_filters, kernel_size = (1,1), activation = None)
        
        
    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        h = self.batch_norm_1(inputs, training = training)
        h = tf.nn.relu(h)
        h = self.conv_2D_1(h)
        h = self.batch_norm_2(h, training = training)
        h = tf.nn.relu(h)
        h = self.conv_2D_2(h)
        output = tf.add(h, self.conv_2D_3(inputs))
        return output



# Test your custom layer - the following should create a model using your layer

test_model = tf.keras.Sequential([FiltersChangeResidualBlock(16, input_shape=(32, 32, 3), name="fc_resnet_block")])
test_model.summary()








#### GRADED CELL ####

# Complete the following class. 
# Make sure to not change the class or method names or arguments.

class ResNetModel(Model):

    def __init__(self, **kwargs):
        """
        The class initialiser should call the base class initialiser, passing any keyword
        arguments along. It should also create the layers of the network according to the
        above specification.
        """
        super(ResNetModel, self).__init__(**kwargs)
        self.conv_2D_1 = Conv2D(32, (7,7), strides = 2)
        self.residualBlock = ResidualBlock()
        self.conv_2D_2 = Conv2D(32, (3,3), strides = 2)
        self.filtersChangeResidualBlock = FiltersChangeResidualBlock(64)
        self.flatten = Flatten()
        self.dense = Dense(10, activation = 'softmax')        
        
    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the initialiser.
        """
        h = self.conv_2D_1(inputs)
        h = self.residualBlock(h, training = training)
        h = self.conv_2D_2(h)
        h = self.filtersChangeResidualBlock(h, training = training)
        h = self.flatten(h)
        h = self.dense(h)
        return h
        
        



# Create the model

resnet_model = ResNetModel()






# Create the optimizer and loss

optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()






#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

@tf.function
def grad(model, inputs, targets, loss):
    """
    This function should compute the loss and gradients of your model, corresponding to
    the inputs and targets provided. It should return the loss and gradients.
    """
    with tf.GradientTape() as tape:
        grad_loss = loss(targets, model(inputs))
        grads = tape.gradient(grad_loss, model.trainable_variables)
    return grad_loss, grads






#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def train_resnet(model, num_epochs, dataset, optimizer, loss, grad_fn):
    """
    This function should implement the custom training loop, as described above. It should 
    return a tuple of two elements: the first element is a list of loss values per epoch, the
    second is a list of accuracy values per epoch
    """
    train_loss_output = []
    train_acc_output = []
    
    for epoch in range(num_epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for x, y in train_dataset:
            loss_, grads = grad(model, x, y, loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss(loss_)
            epoch_acc((y), model(x))
            
        train_loss_output.append(epoch_loss.result())
        train_acc_output.append(epoch_acc.result())
        
    return train_loss_output, train_acc_output
    
    
    


# Train the model for 8 epochs

train_loss_results, train_accuracy_results = train_resnet(resnet_model, 8, train_dataset, optimizer_obj, 
                                                          loss_obj, grad)







fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

axes[0].set_xlabel("Epochs", fontsize=14)
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_title('Loss vs epochs')
axes[0].plot(train_loss_results)

axes[1].set_title('Accuracy vs epochs')
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()




# Compute the test loss and accuracy

epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

for x, y in test_dataset:
    model_output = resnet_model(x)
    epoch_loss_avg(loss_obj(y, model_output))  
    epoch_accuracy(to_categorical(y), model_output)

print("Test loss: {:.3f}".format(epoch_loss_avg.result().numpy()))
print("Test accuracy: {:.3%}".format(epoch_accuracy.result().numpy()))






# Run this cell to get model predictions on randomly selected test images

num_test_images = test_images.shape[0]

random_inx = np.random.choice(test_images.shape[0], 4)
random_test_images = test_images[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = resnet_model(random_test_images)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.5, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(5., -2., f'Class {label} ({image_labels[label]})')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_xticklabels(image_labels, rotation=0)
    pred_inx = np.argmax(prediction)
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {image_labels[pred_inx]}")
    
plt.show()





































