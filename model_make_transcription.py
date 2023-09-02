import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# バージョン確認
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# EMNISTデータセットをダウンロードして読み込む
X_train, y_train = extract_training_samples('byclass')
X_test, y_test = extract_test_samples('byclass')

# データの正規化
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 画像のリサイズ
new_dim = (128, 128)
X_train = np.array([cv2.resize(img, new_dim) for img in X_train])
X_test = np.array([cv2.resize(img, new_dim) for img in X_test])

# データのリシェイプ
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)

# クラスの数を確認
num_classes = len(np.unique(y_train))

# ターゲット変数をone-hotエンコーディング
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # クラスの数に合わせてアウトプット数を変更

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練とHistoryオブジェクトの取得
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 精度と損失のプロット
plt.figure(figsize=[10,5])

# 精度
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()

# 損失
plt.subplot(122)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

plt.tight_layout()
plt.show()

# モデルの保存
model.save("model_transcription.h5")
