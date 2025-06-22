import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Users\vinhv\OneDrive\Documents\digit_recognition_int8.h5")
# Print the model summary to verify the structure
model.summary()