from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv3D, Dropout, SpatialDropout3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.regularizers import l1, l2, l1_l2

classes = ['Idle', 'Speak']
colormode = 'landmarks'
optical_flow = False

def get_model(input_shape):
    return Sequential([
        InputLayer(input_shape=input_shape),
        Conv3D(16, (4, 3, 3), activation='relu', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="conv3d_0"),
	SpatialDropout3D(0.4),
        MaxPooling3D(),
        Conv3D(32, (3, 3, 3), activation='relu', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="conv3d_1"),
        SpatialDropout3D(0.4),
        MaxPooling3D(),
        Conv3D(64, (1, 3, 3), activation='relu', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="conv3d_2"),
        SpatialDropout3D(0.35),
        MaxPooling3D(),
        Flatten(),
        Dense(256, activation='relu', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="dense_0"),
        Dropout(0.5),
        Dense(128, activation='relu', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="dense_1"),
        Dropout(0.25),
        Dense(2, activation='softmax', bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001), name="softmax")
    ])