import os
import gdown
import cv2
import dlib
import numpy as np
from typing import List
import tensorflow as tf
from collections import defaultdict

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def vocabulary():

    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    return char_to_num, num_to_char

def load_video(path: str, width: int = 64, height: int = 64, detector=detector, predictor=predictor) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            lip_left = landmarks.part(48).x
            lip_right = landmarks.part(54).x
            lip_top = min(landmarks.part(50).y, landmarks.part(51).y)
            lip_bottom = max(landmarks.part(58).y, landmarks.part(59).y)

            lip_frame = frame[lip_top:lip_bottom, lip_left:lip_right]

            lip_frame_resized = cv2.resize(lip_frame, (width, height))

            lip_frame_gray = cv2.cvtColor(lip_frame_resized, cv2.COLOR_BGR2GRAY)

            frames.append(lip_frame_gray)

    cap.release()

    frames = tf.convert_to_tensor(frames, dtype=tf.float32)

    mean = tf.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    frames = (frames - mean) / std

    frames = tf.expand_dims(frames, axis=-1)

    return frames

def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]

    char_to_num , _ = vocabulary()
    token_array = np.array(char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1))), dtype=np.int32)
    return token_array

# funkcija load_data ucitava aligments(labele), video i izdvaja automatski usne na video snimku

def load_data(speaker_num, path: str):
    path = path.numpy()
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    else:
        path = str(path)

    #print(path)
    file_name = os.path.basename(path).split('.')[0]
    video_path = os.path.join('data', f's{speaker_num}', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', f's{speaker_num}', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments