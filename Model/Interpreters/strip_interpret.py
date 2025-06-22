import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.quantize import strip_quantization

with quantize_scope():
    # Load the pre-trained model
    model = tf.keras.models.load_model(r"C:\\Users\\vinhv\\OneDrive\\Documents\\digit_recognition.h5")
    
# Strip the quantization layers
stripped_model = strip_quantization(model)
# Save the stripped model
stripped_model.save(r"C:\\Users\\vinhv\\OneDrive\\Documents\\digit_recognition_stripped.h5")
# print(keras.__version__)
