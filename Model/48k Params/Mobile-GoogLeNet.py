import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow_model_optimization.quantization.keras import quantize_model

def depthwise_separable_conv(x, pointwise_filters, kernel_size, strides=(1,1), padding='same', name=None):
    """
    Một khối DepthwiseSeparable: 
    - DepthwiseConv2D (kernel_size × kernel_size), 
    - BatchNorm + ReLU
    - Pointwise Conv2D (1×1) để kết hợp channel, 
    - BatchNorm + ReLU
    """
    # Depthwise
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=f"{name}_dwconv")(x)
    x = BatchNormalization(name=f"{name}_dw_bn")(x)
    x = Activation('relu', name=f"{name}_dw_relu")(x)
    # Pointwise
    x = Conv2D(pointwise_filters, (1,1), padding='same', use_bias=False, name=f"{name}_pwconv")(x)
    x = BatchNormalization(name=f"{name}_pw_bn")(x)
    x = Activation('relu', name=f"{name}_pw_relu")(x)
    return x

def inception_block(x, f1x1, f3x3_reduce, f3x3, f5x5_reduce, f5x5, fpool_proj, name=None):
    """
    Inception-like module nhưng ở chỗ:
    - Nhánh 3×3 và 5×5 đều dùng DepthwiseSeparableConv để tiết kiệm tham số
    - Nhánh 1×1 vẫn là Conv2D thông thường
    - Nhánh Pooling: MaxPool -> Conv2D(1×1)
    """
    # Nhánh 1×1 convolution
    branch1 = Conv2D(f1x1, (1,1), padding='same', use_bias=False, name=f"{name}_b1_conv")(x)
    branch1 = BatchNormalization(name=f"{name}_b1_bn")(branch1)
    branch1 = Activation('relu', name=f"{name}_b1_relu")(branch1)

    # Nhánh 3×3: 1×1 → Depthwise(3×3) → Pointwise(f3x3)
    branch2 = Conv2D(f3x3_reduce, (1,1), padding='same', use_bias=False, name=f"{name}_b2_conv1x1")(x)
    branch2 = BatchNormalization(name=f"{name}_b2_bn1")(branch2)
    branch2 = Activation('relu', name=f"{name}_b2_relu1")(branch2)
    branch2 = depthwise_separable_conv(branch2, pointwise_filters=f3x3, kernel_size=(3,3), name=f"{name}_b2_ds")

    # Nhánh 5×5: 1×1 → Depthwise(5×5) → Pointwise(f5x5)
    branch3 = Conv2D(f5x5_reduce, (1,1), padding='same', use_bias=False, name=f"{name}_b3_conv1x1")(x)
    branch3 = BatchNormalization(name=f"{name}_b3_bn1")(branch3)
    branch3 = Activation('relu', name=f"{name}_b3_relu1")(branch3)
    branch3 = depthwise_separable_conv(branch3, pointwise_filters=f5x5, kernel_size=(5,5), name=f"{name}_b3_ds")

    # Nhánh Pooling: MaxPool → 1×1 Conv
    branch4 = MaxPooling2D((3,3), strides=(1,1), padding='same', name=f"{name}_b4_pool")(x)
    branch4 = Conv2D(fpool_proj, (1,1), padding='same', use_bias=False, name=f"{name}_b4_conv1x1")(branch4)
    branch4 = BatchNormalization(name=f"{name}_b4_bn")(branch4)
    branch4 = Activation('relu', name=f"{name}_b4_relu")(branch4)

    # Kết hợp (Concatenate) bốn nhánh
    out = Concatenate(name=f"{name}_concat")([branch1, branch2, branch3, branch4])
    return out

def build_model_inception_ds(input_shape=(28,28,1), num_classes=10):
    """
    Xây toàn bộ mô hình với 2 Inception Blocks, 
    mỗi Inception Block dùng DepthwiseSeparableConv ở nhánh 3x3 và 5x5,
    cuối cùng là GlobalAveragePooling2D + Dense(256) + Dropout(0.3) + Dense(num_classes).
    """

    inputs = Input(shape=input_shape, name="input_image")

    # --- Đầu tiên: Conv2D thường để trích đặc trưng ban đầu
    x = Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, name="stem_conv")(inputs)
    x = BatchNormalization(name="stem_bn")(x)
    x = Activation('relu', name="stem_relu")(x)
    x = MaxPooling2D((2,2), name="stem_pool")(x)     # Kết quả: 14x14x16

    # --- Inception Block 1 (đầu vào: 14x14x16)
    #   Chọn thông số cho mỗi nhánh sao cho tổng channels ~88 để giữ mô hình vừa đủ mạnh
    x = inception_block(x, f1x1=16, f3x3_reduce=16, f3x3=24, f5x5_reduce=8, f5x5=16, fpool_proj=16, name="incept1")
    # nhánh 1x1 → 16 filters
    # nhánh 3x3 giảm 16 → depthwise → pointwise 24
    # nhánh 3x3 giảm 16 → depthwise → pointwise 24
    # nhánh 5x5 giảm 8 → depthwise → pointwise 16
    # nhánh pooling → 1x1 conv 16

    # Output sau Inception1: 14x14x(16+24+16+16) = 14x14x72
    x = MaxPooling2D((2,2), name="pool_after_incept1")(x)  # → 7x7x72

    # --- Inception Block 2 (đầu vào: 7x7x72)
    #   Tăng số filter nhánh để mô hình dày hơn một chút
    x = inception_block(x, f1x1=32, f3x3_reduce=24, f3x3=48, f5x5_reduce=12, f5x5=24, fpool_proj=24, name="incept2")
    # nhánh 1x1 → 32
    # nhánh 3x3 giảm 24 → ds → pointwise 48
    # nhánh 5x5 giảm 12 → ds → pointwise 24
    # nhánh pooling → 1x1 conv 24

    # Output sau Inception2: 7x7x(32+48+24+24) = 7x7x128

    # --- Global Average Pooling
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)  # → vector shape (batch_size, 128)

    # --- Dense 256 + Dropout 0.3
    x = Dense(256, activation='relu', name="dense_256")(x)  # → (batch_size, 256)
    x = Dropout(0.3, name="dropout_final")(x)

    # --- Output layer
    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Incept_DS_DigitModel")
    return model

# -----------------------------------------------------------
# KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
# -----------------------------------------------------------
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
   
    model = build_model_inception_ds()
    qat_model = quantize_model(model)
    
    qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    qat_model.summary()
#   qat_model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2)
#   test_loss, test_acc = qat_model.evaluate(x_test, y_test)
#   print("Test Accuracy after QAT:", test_acc)
#   qat_model.save(r"C:\Users\vinhv\OneDrive\Máy tính\digit_recognition.h5")
#   qat_model.save(r"C:\Users\vinhv\OneDrive\Documents\digit_recognition", save_format='tf')

