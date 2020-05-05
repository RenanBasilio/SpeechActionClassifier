# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pathlib
import loader
import dlib
import cv2 as cv2

video = loader.load_video_as_ndarray(pathlib.Path("./data/Train/6-1/Speak/001755.mp4"), color_mode='raw', optical_flow=False)
face_detector = dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")
for frame in video:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(frame, 1)
    print(faces.count)