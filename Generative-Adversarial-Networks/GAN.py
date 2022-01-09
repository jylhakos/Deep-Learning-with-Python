import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import IPython.display as ipd
from IPython.display import IFrame

training=True

def sample_true_distr(n=10000):
    """
    Function to generate samples from the quadratic distribution.
    Takes as input sample size `n` and returns set of n-pairs (x,y),
    where x is a real number and y is a square of x plus a constant.
    """

    # set random seed for reproducibility
    np.random.seed(42)
    # draw samples from normal distribution
    x = 10*(np.random.random_sample((n,))-0.5)
    # compute y
    y = 10 + x*x

    return np.array([x, y]).T

# Create a dataset with samples from the quadratic distribution
target_data = sample_true_distr()

# Plot the first 32 datapoints
plt.scatter(target_data[:32,0],target_data[:32,1])
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.show()

batch_size = 32

def cast_func(ds):
    
    """
    Function that takes tensor as input and
    returns it as a tensor of dtype tf.float32.
    """
    ds = tf.cast(ds, tf.float32)
    print('cast', ds)
    return ds

#print(target_data)
# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(target_data)
print(dataset)
#for element in dataset:
#    print(element)
# map
dataset = dataset.map(lambda x: cast_func(x))
print(dataset)
#for element in dataset:
#    print(element)
# shuffle
dataset = dataset.shuffle(buffer_size=1000, seed=42)
print(dataset)
#for element in dataset:
#    print(element)
# batching
dataset =  dataset.batch(batch_size=batch_size, drop_remainder=True)
print(dataset)
#for element in dataset:
#    print(element)
# prefetch

dataset = dataset.prefetch(buffer_size=1)

print(dataset)

# size of the random vector
codings_size = 10

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu,input_shape=[codings_size]),
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=2) 
], name="Generator")

generator.build(input_shape=(None,codings_size))

generator.summary()

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu, input_shape=[2]),
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)
], name="Discriminator")

discriminator.build(input_shape=(None,codings_size))

discriminator.summary()

gan = tf.keras.models.Sequential([generator, discriminator])

tf.keras.utils.plot_model(
    gan,
    show_shapes=True, 
    show_layer_names=True
)

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

discriminator.trainable = False

gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=90):
    saved_samples = np.zeros((int(n_epochs/10),2,batch_size,2))
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):             
        for X_batch in dataset:
            
            # phase 1 - training the Discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            gen_samples = generator(noise)
            X_fake_and_real = tf.concat([gen_samples, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.train_on_batch(X_fake_and_real, y1)
            
            # phase 2 - training the Generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            gan.train_on_batch(noise, y2)
            
        if epoch%10 == 0:
            print("Epoch {}/{}".format(epoch, n_epochs)) 
            saved_samples[int(epoch/10),0,:,:] = X_batch
            saved_samples[int(epoch/10),1,:,:] = gen_samples

    return saved_samples

if training:                        
    saved_samples = train_gan(gan, dataset, batch_size, codings_size)

if training:
    plt.figure(figsize=(8, 8))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # plot real samples
        plt.scatter(saved_samples[i,0,:,0],saved_samples[i,0,:,1])
        # plot generated (fake) samples
        plt.scatter(saved_samples[i,1,:,0],saved_samples[i,1,:,1])
        plt.axis("off")
    plt.show()
