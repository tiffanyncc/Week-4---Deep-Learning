import tensorflow as tf
import numpy as np

def find_best_learning_rate_with_scheduler(x_train, y_train):
    tf.keras.utils.set_random_seed(42)
    #model = None
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1), # add an extra layer
        tf.keras.layers.Dense(1) # output layer
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["accuracy"])

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )

    history = model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[lr_scheduler])
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    return lrs, history
