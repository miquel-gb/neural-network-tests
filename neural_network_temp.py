import numpy
import tensorflow as tf

# Initializes arrays of values for the training session
celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Initializes a simple neural network with just one dense layer 
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# Sets the learning rate for the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = tf.keras.losses.MeanSquaredError()
)

print('\nFinished initializing data.')
# Trains the model for 1000 epochs
print ('\nStarting model training...')
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('Finished training!')

# Tries to guess a value that wasn't given in the examples with the learnt info
testResult = model.predict([97])
print('\nLets try to guess the temperature of 97 degrees celsius:')
print('The predicted value for 97°C is: ', testResult, '°F, the right value is 206.6°C')

# Displays the used weights and biases for the model
print('\nThe used weights and biases for the model are:')
print(layer.get_weights())