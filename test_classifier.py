import cv2
import pickle
import numpy as np
import mediapipe as mp

model_dict = pickle.load(open('model.classify','rb'))
model = model_dict['model']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = {0 : 'V', 1: 'Thumbs up', 2: 'Thumbs Down', 3: 'B', 4: 'C', 
          5 : 'Super', 6: 'Baba', 7: 'Gun', 8 : 'Toilet', 9: 'Five'}

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data_arr = []
    x_ =[]
    y_ =[]
    H, W, _ = frame.shape
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            mx = min(x_)
            my = min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_arr.append(x)
                data_arr.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        if(len(data_arr) != 42):
            continue
        prediction = model.predict([np.asarray(data_arr)])
        predicted_character = labels[int(prediction[0])]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame, predicted_character, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0 , 0), 3,
                    cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()