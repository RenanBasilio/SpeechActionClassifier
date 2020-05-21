from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv3D, Dropout, MaxPooling3D, AveragePooling3D

classes = ['Idle', 'Speak']
colormode = 'landmarks'
optical_flow = False

def get_model(input_shape):
    return Sequential([
        InputLayer(input_shape=input_shape),
        Conv3D(16, (4, 3, 3), activation='relu'),
        MaxPooling3D(),
        Conv3D(32, (3, 3, 3), activation='relu'),
        MaxPooling3D(),
        Conv3D(64, (1, 3, 3), activation='relu'),
        MaxPooling3D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])