import cv2
import numpy as np


cap = cv2.VideoCapture(0)
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lk_params = dict(winSize=(50, 50),
                 maxLevel=20,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# mouse function


def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


cv2.namedWindow('frame', )
cv2.setMouseCallback('frame', select_point)
point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected:
        cv2.circle(frame, point, 5, (0, 0, 225), 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None,  **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points
        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
