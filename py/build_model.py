from tensorflow.keras import layers

#Build architecture

def fit_model(X_train, y_train, num_freq_samples, num_epochs):
    model = tf.keras.Sequential()
    model.add(layers.SimpleRNN((10), batch_input_shape=(None,398, NUM_FREQ_SAMPLES), return_sequences=False))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=num_epochs)

    return model, hist
