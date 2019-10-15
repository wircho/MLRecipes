# KERAS XCEPTION FINE TUNING
# Fine tuning the Xception convolutional neural network (CNN),
# pre-trained on ImageNet, to create an arbitrary image
# classification model.

import os
import math
import pandas
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_weights_path = # Where do you want to save your model weights (ending in .hdf5)

### 1. Choose your data:

# Assuming all your images are in the same directory:
directory = # Path to the images folder
names = # All your training AND validation image names.
labels = # Labels for each of the names above (len(names) must be == len(labels)).
classes = # All unique labels (without repetition) in your preferred order,
          # e.g. ["cat", "dog", "mouse"]
n = len(classes) # Number of unique classes - useful later.

###
### 2. Split the data (training + validation):
###

train_perc = 0.8 # Choose your split.
train_count = math.floor(train_perc * len(names))

train_names = names[:train_count]
train_labels = labels[:train_count]
val_names = names[train_count:]
val_labels = labels[train_count:]

###
### 3. Load training data (as a generator with random transforms):
###

# Tweak parameters below as needed
generator = keras.preprocessing.image.ImageDataGenerator(
	rotation_range = 25,
	width_shift_range = 0.225,
	height_shift_range = 0.25,
	horizontal_flip = True,
	brightness_range = (0.3, 1.7),
	fill_mode = "nearest",
	preprocessing_function = keras.applications.xception.preprocess_input
)
# Create generator
batch_size = 16 # Tweak if needed
train_flow = generator.flow_from_dataframe(
	pandas.DataFrame({"filename": train_names, "class": train_labels}, index: range(len(train_names))),
	x_col = "filename",
	y_col = "class",
	classes = classes,
	directory = directory,
	target_size = (299, 299),
	batch_size = batch_size
)

###
### 4. Load validation data (as an array):
###

# Load val_x using Keras preprocessing functions
val_x = np.array([img_to_array(load_img(os.path.join(directory, name), target_size = (299, 299))) for name in val_names])
val_x = keras.applications.xception.preprocess_input(val_x)
# Load val_y by converting val_labels to a float array
val_y = tools.main.numpy.array([[1.0 if clss == label else 0.0 for clss in classes] for label in val_labels])

###
### 5. Load the Xception CNN, pre-trained on ImageNet, and
###    trimmed so that the "top" (1000 labels) is not present:
###

# xception: (299, 299, 3) -> (10, 10, 2048)
xception = keras.applications.xception.Xception(weights = "imagenet", include_top = False, input_shape = (299, 299, 3))

###
### 6. Create the new "top" with n labels:
###

# top: (10, 10, 2048) -> (n,)
top = keras.models.Sequential()
top.add(keras.layers.GlobalAveragePooling2D(input_shape = (10, 10, 2048)))
top.add(keras.layers.Dense(512))
top.add(keras.layers.BatchNormalization())
top.add(keras.layers.Activation("relu"))
top.add(keras.layers.Dense(n))
top.add(keras.layers.BatchNormalization())
top.add(keras.layers.Activation("softmax"))

###
### 7. Create the model by combining xception and top:
###

# model: (299, 299, 3) -> (n,)
model = keras.models.Sequential()
model.add(xception)
model.add(top)

###
### 8. Freeze the first few layers (so they do not change during training)
###

num_frozen_layers = 65 # Tweak if needed
for layer in xception.layers[:num_frozen_layers]: layer.trainable = False
for layer in xception.layers[num_frozen_layers:]: layer.trainable = True

###
### 9. Randomize the last few layers' weights (reduces overfitting)
###

def randomize(model, first_layer_index = 0):
	session = keras.backend.get_session()
	for layer in model.layers[first_layer_index:]:
		if isinstance(layer, tf.keras.models.Model):
			randomize(layer)
			continue
		for value in layer.__dict__.values():
			if not hasattr(value, "initializer"): continue
			value.initializer.run(session = session)

first_randomized_layer_index = 115  # Tweak if needed
randomize(xception, first_layer_index = first_randomized_layer_index)

###
### 10. Compile model:
###

model.compile(
	loss: "categorical_crossentropy",
	optimizer: keras.optimizers.SGD(lr: 0.0, momentum: 0.5, decay: 0, nesterov: False),
	metrics: ["accuracy"]
)

###
### 11. Train model:
###

learning_rates = [0.08 * 0.8 ** math.floor(i / 2) for i in range(20)] # Tweak if needed
steps = math.ceil(len(train_names) / batch_size)
scheduler = keras.callbacks.LearningRateScheduler(lambda i: learning_rates[i])
checkpoint = keras.callbacks.ModelCheckpoint(model_weights_path, verbose: 1, save_best_only: True)
model.fit_generator(
	train_flow,
	steps,
	validation_data: (val_x, val_y),
	epochs: len(learning_rates),
	callbacks: [scheduler, checkpoint],
	verbose: 2
)

###
### 12. Loading your model at a later stage:
###

# Follow steps 5,6,7, and then:
# model.load_weights(model_weights_path)





