import tensorflow as tf
import tensorflow_model_optimization as tfmot

# converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\vinhv\OneDrive\Documents\digit_recognition")
# tflite_model = converter.convert()
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# with open(r"C:\Users\vinhv\OneDrive\Documents\digit_recognition_full_power.tflite", "wb") as f:
#    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=r"C:\Users\vinhv\OneDrive\Documents\digit_recognition_full_power.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input details", input_details)
print("ouput details", output_details)