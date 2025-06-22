import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"C:\Users\vinhv\OneDrive\Documents\digreg_10k_float32.h5")
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)  

def representative_dataset():
    for i in range(1000):
        yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]    
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(r"C:\Users\vinhv\OneDrive\Documents\digreg_10k_int8.tflite", 'wb') as f:
    f.write(tflite_model)
#   print(model.summary())

