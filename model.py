from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv3D, Dropout, SpatialDropout3D, MaxPooling3D, AveragePooling3D

classes = ['Idle', 'Speak']
colormode = 'landmarks'
optical_flow = False

def get_model(input_shape):
    return Sequential([
        InputLayer(input_shape=input_shape),
        Conv3D(16, (1, 3, 3), activation='relu'),
        MaxPooling3D((1,2,2)),
        Conv3D(24, (3, 3, 3), activation='relu'),
        Conv3D(48, (5, 7, 7), activation='relu'),
        MaxPooling3D((1,2,2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        AveragePooling3D((2,1,1)),
        MaxPooling3D((1,2,2)),
        SpatialDropout3D(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='sigmoid'),
        Dense(2, activation='softmax')
    ])