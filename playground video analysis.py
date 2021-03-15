import cv2
import numpy as np


cap = cv2.VideoCapture("vibration.avi")
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('vibration.avi', fourcc, 20.0, (640, 480))


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame[0])
    # frame = np.fft.fft(frame[0:, ])
    if (cv2.waitKey(1) and 0xFF == ord('q')) or ret is False:
        break
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
