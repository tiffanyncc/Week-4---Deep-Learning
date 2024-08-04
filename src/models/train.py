import tensorflow as tf

def train_neural_network(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result

def train_with_more_epochs(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=100, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result

def train_with_extra_layer(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result

def train_with_more_neurons(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2), # more neurons in hidden layer
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result

def train_with_new_learning_rate(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1), # add an extra layer
        tf.keras.layers.Dense(1) # output layer
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result

def train_with_activation_function(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1), # can also try other activation function
        tf.keras.layers.Dense(1, activation='sigmoid') # output layer with sigmoid activation
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, verbose=0)
    result = model.evaluate(x_train, y_train)
    return model, history, result
