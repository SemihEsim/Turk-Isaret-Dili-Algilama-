import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

# --- AYARLAR ---
SEQUENCE_LENGTH = 32 
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST = ["abla", "bayrak", "ev", "fil", "hastane", "kitap", "misafir", "para", "saat", "yarin"]
NUM_CLASSES = len(CLASSES_LIST)

def frames_extraction(video_path):
    """Sadece HAREKETİ (Kare farklarını) çıkaran fonksiyon"""
    frames_diff_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fark almak için SEQUENCE_LENGTH + 1 kare okumalıyız
    skip_frames_window = max(int(video_frames_count / (SEQUENCE_LENGTH + 1)), 1)
    
    temp_grayscale_frames = []
    for i in range(SEQUENCE_LENGTH + 1):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames_window)
        success, frame = video_reader.read()
        if not success: break
        
        # Gri tonlamaya çevir (Renk, hareket için gereksizdir ve ezberlemeyi artırır)
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        temp_grayscale_frames.append(gray_frame)
    
    video_reader.release()

    # KARE FARKI HESAPLA: (Şu anki kare - Önceki kare)
    for j in range(1, len(temp_grayscale_frames)):
        diff = cv2.absdiff(temp_grayscale_frames[j], temp_grayscale_frames[j-1])
        # Normalizasyon (0-1)
        normalized_diff = diff / 255.0
        # 3D CNN 3 kanal beklediği için derinlik ekle
        rgb_diff = np.stack((normalized_diff,)*3, axis=-1)
        frames_diff_list.append(rgb_diff)
        
    return frames_diff_list

def create_dataset(dataset_dir):
    features = []; labels = []
    for class_index, class_name in enumerate(CLASSES_LIST):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir): continue
        print(f"Isleniyor: {class_name}")
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            frames = frames_extraction(video_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
    return np.asarray(features), np.array(labels)

def create_motion_3dcnn():
    model = Sequential()
    # Zaman boyutunda (5 karelik) geniş tarama yapan katman
    model.add(Conv3D(32, kernel_size=(5, 3, 3), activation='relu', padding='same',
                     input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6)) # Ezberlemeyi önlemek için yüksek dropout
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- ANA AKIŞ ---
print("Veriler hazirlaniyor (Sadece Hareket Odakli)...")
X_train, y_train = create_dataset('archive/train')
X_val, y_val = create_dataset('archive/val')

model = create_motion_3dcnn()
print("Egitim basliyor...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8)

model.save('motion_3dcnn_model.h5')
print("Model kaydedildi.")