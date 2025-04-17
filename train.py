import numpy as np
import os
import cv2
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import Input, TimeDistributed, BatchNormalization, Dropout, Flatten, LSTM, Bidirectional, Dense, Input # type: ignore
from keras.applications.mobilenet_v2 import MobileNetV2 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import accuracy_score
# Set dataset path
DATASET_DIR = "Violence detection\Dataset"
CLASSES_LIST = ["NonViolence", "Violence"]
IMAGE_HEIGHT, IMAGE_WIDTH = 96,96
SEQUENCE_LENGTH = 16

# Function to extract frames
def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

# Function to create dataset
def create_dataset(): 
    features, labels = [], []
    for class_index, class_name in enumerate(CLASSES_LIST):
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
    return np.array(features), np.array(labels)

# Load dataset
features, labels = create_dataset()
np.save("features.npy", features)
np.save("labels.npy", labels)
features, labels = np.load("features.npy"), np.load("labels.npy")
labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, shuffle=True, random_state=42)

# Load MobileNetV2 model
mobilenet = MobileNetV2(weights= "imagenet", include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
mobilenet.trainable = True
for layer in mobilenet.layers[:-40]:
    layer.trainable = False

# Function to create the model
def create_model():
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Flatten()))

    lstm_fw = LSTM(units=64, kernel_regularizer=l2(0.01))  # Regularization added here
    lstm_bw = LSTM(units=64, go_backwards=True, kernel_regularizer=l2(0.01))
    
    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  # Regularization added here
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu',  kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu',  kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu',  kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    return model

# Train the model
MoBiLSTM_model = create_model()
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=40, 
    restore_best_weights=True, 
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.6, 
    patience=10, 
    min_lr=0.00005, 
    verbose=1
)
MoBiLSTM_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
MoBiLSTM_model.fit(features_train, labels_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.2,callbacks=[early_stopping, reduce_lr])

#accuracy
labels_predict = MoBiLSTM_model.predict(features_test)
labels_predict = np.argmax(labels_predict , axis=1)
labels_test_normal = np.argmax(labels_test , axis=1)
AccScore = accuracy_score(labels_predict, labels_test_normal)
print(f"Accuracy: {AccScore * 100:.2f}%")

# Save the trained model
model_path = "violence_detection_model_.keras"
MoBiLSTM_model.save(model_path,save_format="keras") 
print(f"Model saved at {model_path}")
