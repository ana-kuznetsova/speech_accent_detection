import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Build architecture

def fit_model(X_train, y_train, X_test, y_test, num_freq_samples, num_epochs):
	

	model = tf.keras.Sequential()
	model.add(layers.LSTM((512), batch_input_shape=(None,None,num_freq_samples), return_sequences=False))
	model.add(layers.Dense(2, activation="softmax"))
	#model.add(layers.Dense(2, activation = "softmax"))
	
	#model.add(Conv2D(40 , (3,3) , input_shape = (359, num_freq_samples,1) ))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size= (2,2)))
	#model.add(Conv2D(20 , (3,3) ))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size= (2,2)))
	#model.add(Flatten())
	#model.add(Dense(2, activation = "softmax"))
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size =24)
	return model, hist
