import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation, DepthwiseConv2D, 
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.models import Model

def depthwise_separable_conv(x, pointwise_filters, kernel_size, strides=(1,1), padding='same', name=None):
    # Depthwise
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=f"{name}_dwconv")(x)
    x = BatchNormalization(name=f"{name}_dw_bn")(x)
    x = Activation('relu', name=f"{name}_dw_relu")(x)
    # Pointwise
    x = Conv2D(pointwise_filters, (1,1), padding='same', use_bias=False, name=f"{name}_pwconv")(x)
    x = BatchNormalization(name=f"{name}_pw_bn")(x)
    x = Activation('relu', name=f"{name}_pw_relu")(x)
    return x

def inception_block(x, f1x1, f3x3_reduce, f3x3, fpool_proj, name=None):
    branch1 = Conv2D(f1x1, (1,1), padding='same', use_bias=False, name=f"{name}_b1_conv")(x)
    branch1 = BatchNormalization(name=f"{name}_b1_bn")(branch1)
    branch1 = Activation('relu', name=f"{name}_b1_relu")(branch1)

    branch2 = Conv2D(f3x3_reduce, (1,1), padding='same', use_bias=False, name=f"{name}_b2_conv1x1")(x)
    branch2 = BatchNormalization(name=f"{name}_b2_bn1")(branch2)
    branch2 = Activation('relu', name=f"{name}_b2_relu1")(branch2)
    branch2 = depthwise_separable_conv(branch2, pointwise_filters=f3x3, kernel_size=(3,3), name=f"{name}_b2_ds")

    branch3 = MaxPooling2D((3,3), strides=(1,1), padding='same', name=f"{name}_b3_pool")(x)
    branch3 = Conv2D(fpool_proj, (1,1), padding='same', use_bias=False, name=f"{name}_b3_conv1x1")(branch3)
    branch3 = BatchNormalization(name=f"{name}_b3_bn")(branch3)
    branch3 = Activation('relu', name=f"{name}_b3_relu")(branch3)
    out = Concatenate(name=f"{name}_concat")([branch1, branch2, branch3])
    return out

def create_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Khối đầu vào
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool1')(x)
    
    # Khối Inception đầu tiên
    x = inception_block(x, f1x1=16, f3x3_reduce=16, f3x3=24, fpool_proj=16, name='inception_block1')
    
    # Khối Inception thứ hai
    x = inception_block(x, f1x1=32, f3x3_reduce=32, f3x3=48, fpool_proj=32, name='inception_block2')
    
    # Toàn cục Average Pooling và đầu ra
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(64, activation='relu', name='dense_layer')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    x = Dense(32, activation='relu', name='dense_layer_2')(x)

    # Đầu ra cuối cùng
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == "__main__":
    model = create_model()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2)
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print("Test Accuracy:", test_acc)

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='train_accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()
    # model.summary()
    # Lưu mô hình nếu cần
    model.save(r"C:\Users\vinhv\OneDrive\Documents\digreg_20k.h5")
