# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Add this line
# width, height = 128, 128  # You can use the same values you used during training
#
# test_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test'
#
# # Create an ImageDataGenerator for normalization (no augmentation for test data)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # Create a flow_from_directory generator for test data
# test_generator = test_datagen.flow_from_directory(
#     test_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='sparse',  # Assuming your test labels are not one-hot encoded
#     shuffle=False  # No need to shuffle for test data
# )
#
# # Evaluate the model on the test data
# eval_result = model.evaluate(test_generator)
#
# # Print the evaluation result (accuracy and loss)
# print("Test Accuracy:", eval_result[1])
# print("Test Loss:", eval_result[0])
#
# # Make predictions on the test data
# predictions = model.predict(test_generator)
#
# # Print the predicted class indices for the first few samples
# print("Predicted Class Indices:")
# print(np.argmax(predictions, axis=1)[:10])

# import numpy as np
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Define the target width and height for image resizing
# width, height = 128, 128
#
# # Load the trained model
# model = keras.models.load_model('image_classifier_model.h5')  # Adjust the filename accordingly
#
# # Assuming you have a separate folder for test data
# test_directory = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test'
#
# # Create an ImageDataGenerator for normalization (no augmentation for test data)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # Create a flow_from_directory generator for test data
# test_generator = test_datagen.flow_from_directory(
#     test_directory,
#     target_size=(width, height),
#     batch_size=32,
#     class_mode='sparse',  # Assuming your test labels are not one-hot encoded
#     shuffle=False  # No need to shuffle for test data
# )
#
# # Load the trained model
# model = keras.models.load_model('image_classifier_model.h5')  # Adjust the filename accordingly
#
# # Evaluate the model on the test data
# eval_result = model.evaluate(test_generator)
#
# # Print the evaluation result (accuracy and loss)
# print("Test Accuracy:", eval_result[1])
# print("Test Loss:", eval_result[0])
#
# # Make predictions on the test data
# predictions = model.predict(test_generator)
#
# # Print the predicted class indices for the first few samples
# print("Predicted Class Indices:")
# print(np.argmax(predictions, axis=1)[:10])

# from keras.preprocessing import image
# import numpy as np
#
# # Load the trained model
# from keras.models import load_model
# model = load_model('C:\Users\amanm\OneDrive\Desktop\SE project\image_classifier_model.h5')  # Replace with the actual path to your trained model file
#
# # Directory containing test images
# test_data_dir = 'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test\0\im_1.png'  # Replace with the actual path to your test data
#
# # Dimensions of your images
# img_width, img_height = 150, 150
#
# # Load and preprocess each test image
# img_paths = ['path/to/test_data/image1.jpg', 'path/to/test_data/image2.jpg', ...]  # Replace with actual image paths
#
# for img_path in img_paths:
#     img = image.load_img(img_path, target_size=(img_width, img_height))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.  # Normalize the image
#
#     # Make predictions
#     prediction = model.predict(img_array)
#
#     # Print the result
#     if prediction[0][0] > 0.5:
#         print(f"{img_path}: Predicted class - Class 1")
#     else:
#         print(f"{img_path}: Predicted class - Class 0")

# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
#
# # Load the trained model
# model = load_model(r'C:\Users\amanm\OneDrive\Desktop\SE project\image_classifier_model.h5')
#
# # Directory containing test images
# test_data_dir = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test'
#   # Replace with the actual path to your test data
#
# # Dimensions of your images
# img_width, img_height = 150, 150
#
# # Create an ImageDataGenerator for test data
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical',  # Change to 'binary' if you have two classes
#     shuffle=False  # Important: Do not shuffle the test data
# )
#
# # Make predictions on the test data
# predictions = model.predict(test_generator)
#
# # Get class labels from the generator
# class_labels = list(test_generator.class_indices.keys())
#
# # Assign the predicted class labels
# predicted_labels = [class_labels[np.argmax(pred)] for pred in predictions]
#
# # Get the true class labels
# true_labels = test_generator.classes
# true_class_labels = [class_labels[label] for label in true_labels]
#
# # Print the results
# for i in range(len(predicted_labels)):
#     print(f"True label: {true_class_labels[i]}, Predicted label: {predicted_labels[i]}")

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load the model
model = load_model(r'C:\Users\amanm\OneDrive\Desktop\SE project\image_classifier_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# Specify the path to your test data
test_data_dir = r'C:\Users\amanm\OneDrive\Desktop\SE project\ARDIS\Test'
your_target_width = 1000
your_target_height = 1000
your_batch_size = 1000
# Create an ImageDataGenerator for the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(your_target_width, your_target_height),
    batch_size=your_batch_size,
    class_mode='categorical',
    shuffle=False
)

# Check if any images are found
if test_generator.samples == 0:
    print("No images found in the test directory.")
else:
    # Make predictions
    predictions = model.predict(test_generator)
