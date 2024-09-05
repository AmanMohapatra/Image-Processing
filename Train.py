# import os
# from PIL import Image
# import numpy as np

# # Define the directories for the train and test datasets
# test_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test\0'
# train_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Train\0'

# # Function to load and preprocess images from a directory
# def load_images_from_directory(directory):
#     images = []
#     labels = []
#     class_names = sorted(os.listdir(directory))  # Get class names from folder names

#     for class_name in class_names:
#         class_directory = os.path.join(directory, class_name)
#         for filename in os.listdir(class_directory):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 img = Image.open(os.path.join(class_directory, filename))
#                 img = img.resize((width, height))  # Resize the image to a common size
#                 img_array = np.array(img)
#                 images.append(img_array)
#                 labels.append(class_name)

#     return np.array(images), np.array(labels)

# # Set the desired width and height for image resizing
# width, height = 128, 128  # Adjust as needed

# # Load and preprocess the train and test datasets
# X_train, y_train = load_images_from_directory(train_directory)
# X_test, y_test = load_images_from_directory(test_directory)

# # Now you can use X_train, X_test, y_train, and y_test for classification

# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define the main directory containing subdirectories for each class
# main_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Train\0'

# # Set the desired width and height for image resizing
# width, height = 128, 128

# # Create an ImageDataGenerator for data augmentation and normalization
# datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalize pixel values to be between 0 and 1
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# # Create a flow_from_directory generator for training data
# train_generator = datagen.flow_from_directory(
#     main_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='categorical',  # Assumes one-hot encoded labels
#     shuffle=True
# )

# # Define a simple CNN model
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your dataset
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, epochs=10)

# # Save the trained model (optional)
# model.save('image_classifier_model.h5')

# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define the main directory containing subdirectories for each class
# main_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Train'

# # Count the number of subdirectories (classes)
# num_classes = len(os.listdir(main_directory))

# # Set the desired width and height for image resizing
# width, height = 128, 128

# # Create an ImageDataGenerator for data augmentation and normalization
# datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalize pixel values to be between 0 and 1
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# # Create a flow_from_directory generator for training data
# train_generator = datagen.flow_from_directory(
#     main_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='categorical',  # Assumes one-hot encoded labels
#     shuffle=True
# )

# # Define a simple CNN model
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your dataset
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, epochs=10)

# # Save the trained model (optional)
# model.save('image_classifier_model.h5')

# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Define the main directory containing subdirectories for each class
# main_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project'
#
# # Count the number of subdirectories (classes)
# num_classes = len(os.listdir(main_directory))
#
# # Check the number of classes
# print("Number of classes:", num_classes)
#
# # Set the desired width and height for image resizing
# width, height = 128, 128
#
# # Create an ImageDataGenerator for data augmentation and normalization
# datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalize pixel values to be between 0 and 1
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# # Create a flow_from_directory generator for training data
# train_generator = datagen.flow_from_directory(
#     main_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='categorical',  # Assumes one-hot encoded labels
#     shuffle=True
# )
#
# # Check the number of images and classes in the generator
# print("Number of images in the generator:", len(train_generator.filenames))
# print("Number of classes in the generator:", len(train_generator.class_indices))
#
# # Define a simple CNN model
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your dataset
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(train_generator, epochs=10)
#
# # Save the trained model (optional)
# model.save('image_classifier_model.h5')
#
# import os
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Define the main directory containing subdirectories for each class
# main_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project'
#
# # Count the number of subdirectories (classes)
# num_classes = len(os.listdir(main_directory))
#
# # Check the number of classes
# print("Number of classes:", num_classes)
#
# # Set the desired width and height for image resizing
# width, height = 128, 128
#
# # Create an ImageDataGenerator for data augmentation and normalization
# datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalize pixel values to be between 0 and 1
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# # Create a flow_from_directory generator for training data
# train_generator = datagen.flow_from_directory(
#     main_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='categorical',  # Assumes one-hot encoded labels
#     shuffle=True
# )
#
# # Check the number of images and classes in the generator
# print("Number of images in the generator:", len(train_generator.filenames))
# print("Number of classes in the generator:", len(train_generator.class_indices))
#
# # # Define a simple CNN model
# # model = keras.Sequential([
# #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
# #     layers.MaxPooling2D((2, 2)),
# #     layers.Conv2D(64, (3, 3), activation='relu'),
# #     layers.MaxPooling2D((2, 2)),
# #     layers.Flatten(),
# #     layers.Dense(64, activation='relu'),
# #     layers.Dense(num_classes, activation='softmax')  # Update to match the number of classes
# # ])
# #
# # # Compile the model
# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #
# # # Train the model
# # model.fit(train_generator, epochs=10)
#
# # Define a simple CNN model
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3, activation='softmax')  # Update to match the number of classes
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(train_generator, epochs=10)
#
# # Save the trained model (optional)
# model.save('image_classifier_model.h5')

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the main directory containing subdirectories for each class
main_directory = r'C:\Tech\SE project\ARDIS'

# Count the number of subdirectories (classes)
num_classes = len(os.listdir(main_directory))

# Check the number of classes
print("Number of classes:", num_classes)

# Set the desired width and height for image resizing
width, height = 128, 128

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to be between 0 and 1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a flow_from_directory generator for training data
train_generator = datagen.flow_from_directory(
    main_directory,
    target_size=(width, height),
    batch_size=32,
    class_mode='sparse',  # Assumes one-hot encoded labels
    shuffle=True
)

# Check the number of images and classes in the generator
print("Number of images in the generator:", len(train_generator.filenames))
print("Number of classes in the generator:", len(train_generator.class_indices))

# Define a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your dataset
])

# Compile the model using sparse categorical crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the trained model (optional)
model.save('image_classifier_model.h5')
