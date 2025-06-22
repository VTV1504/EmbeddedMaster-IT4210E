import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Concatenate
)
from tensorflow.keras.models import Model

def depthwise_separable_conv(x, pointwise_filters, kernel_size, strides=(1,1), padding='same', name=None):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=f"{name}_dwconv")(x)
    x = BatchNormalization(name=f"{name}_dw_bn")(x)
    x = Activation('relu', name=f"{name}_dw_relu")(x)

    x = Conv2D(pointwise_filters, (1,1), padding='same', use_bias=False, name=f"{name}_pwconv")(x)
    x = BatchNormalization(name=f"{name}_pw_bn")(x)
    x = Activation('relu', name=f"{name}_pw_relu")(x)
    return x

def inception_block(x, f1x1, f3x3_reduce, f3x3, f5x5_reduce, f5x5, fpool_proj, name=None):
    branch1 = Conv2D(f1x1, (1,1), padding='same', use_bias=False, name=f"{name}_b1_conv")(x)
    branch1 = BatchNormalization(name=f"{name}_b1_bn")(branch1)
    branch1 = Activation('relu', name=f"{name}_b1_relu")(branch1)

    branch2 = Conv2D(f3x3_reduce, (1,1), padding='same', use_bias=False, name=f"{name}_b2_conv1x1")(x)
    branch2 = BatchNormalization(name=f"{name}_b2_bn1")(branch2)
    branch2 = Activation('relu', name=f"{name}_b2_relu1")(branch2)
    branch2 = depthwise_separable_conv(branch2, pointwise_filters=f3x3, kernel_size=(3,3), name=f"{name}_b2_ds")

    branch3 = Conv2D(f5x5_reduce, (1,1), padding='same', use_bias=False, name=f"{name}_b3_conv1x1")(x)
    branch3 = BatchNormalization(name=f"{name}_b3_bn1")(branch3)
    branch3 = Activation('relu', name=f"{name}_b3_relu1")(branch3)
    branch3 = depthwise_separable_conv(branch3, pointwise_filters=f5x5, kernel_size=(5,5), name=f"{name}_b3_ds")

    branch4 = MaxPooling2D((3,3), strides=(1,1), padding='same', name=f"{name}_b4_pool")(x)
    branch4 = Conv2D(fpool_proj, (1,1), padding='same', use_bias=False, name=f"{name}_b4_conv1x1")(branch4)
    branch4 = BatchNormalization(name=f"{name}_b4_bn")(branch4)
    branch4 = Activation('relu', name=f"{name}_b4_relu")(branch4)

    out = Concatenate(name=f"{name}_concat")([branch1, branch2, branch3, branch4])
    return out

def build_model_inception_ds(input_shape=(28,28,1), num_classes=10):
    inputs = Input(shape=input_shape, name="input_image")

    x = Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, name="stem_conv")(inputs)
    x = BatchNormalization(name="stem_bn")(x)
    x = Activation('relu', name="stem_relu")(x)
    x = MaxPooling2D((2,2), name="stem_pool")(x)     

    x = inception_block(x, f1x1=16, f3x3_reduce=16, f3x3=24, f5x5_reduce=8, f5x5=16, fpool_proj=16, name="incept1")
    x = MaxPooling2D((2,2), name="pool_after_incept1")(x)  

    x = inception_block(x, f1x1=32, f3x3_reduce=24, f3x3=48, f5x5_reduce=12, f5x5=24, fpool_proj=24, name="incept2")
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)  

    x = Dense(256, activation='relu', name="dense_256")(x)  
    x = Dropout(0.3, name="dropout_final")(x)

    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Incept_DS_DigitModel")
    return model

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model = build_model_inception_ds()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#   history = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2)
#   test_loss, test_acc = model.evaluate(x_test, y_test)
#   print("Test Accuracy:", test_acc)

#    plt.figure(figsize=(12, 4))
#    plt.subplot(1, 2, 1)
#    plt.plot(history.history['accuracy'], label='train_accuracy')
#    plt.plot(history.history['val_accuracy'], label='val_accuracy')
#    plt.title('Training and Validation Accuracy')
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.legend()

#    plt.subplot(1, 2, 2)
#    plt.plot(history.history['loss'], label='train_loss')
#    plt.plot(history.history['val_loss'], label='val_loss')
#    plt.title('Training and Validation Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.legend()

#    plt.show()
    model.save(r"C:\Users\vinhv\OneDrive\Máy tính\digit_recognition_v2.h5")