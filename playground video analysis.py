import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("vibration.avi")
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('vibration.avi', fourcc, 20.0, (640, 480))


def highpass_filter(frame):
    rows, cols = frame.shape
    center_row, center_col = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 60
    center = [center_row, center_col]
    x, y = np.ogrid[: rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    return mask


def lowpass_filter(frame):
    rows, cols = frame.shape
    center_row, center_col = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 250
    center = [center_row, center_col]
    x, y = np.ogrid[: rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    return mask

a = 0
stop = True
while True:
    ret, frame = cap.read()
    if (cv2.waitKey(1) and 0xFF == ord('q')) or ret is False:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_fft = cv2.dft(np.float32(frame_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    frame_shift = np.fft.fftshift(frame_fft)
    high_filter = highpass_filter(frame_gray)
    low_filter = lowpass_filter(frame_gray)
    filtered_frame = frame_shift * low_filter
    filtered_frame = filtered_frame * high_filter
    filter = 20 * np.log(cv2.magnitude(filtered_frame[:, :, 0], filtered_frame[:, :, 1]))
    filter = np.asanyarray(filter, dtype=np.uint8)
    phase_spectrum = np.angle(filtered_frame)[:, :, 0]

    # reversing
    reverse_shift = np.fft.ifftshift(filtered_frame)
    reverse_image = cv2.idft(reverse_shift)
    result = 20 * np.log(cv2.magnitude(reverse_image[:, :, 0], reverse_image[:, :, 1]))
    result = cv2.normalize(result, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # amplitude of the result image
    result_amplitude = 20 * np.log(cv2.magnitude(filtered_frame[:, :, 0], filtered_frame[:, :, 1]))
    result_amplitude = np.asanyarray(result_amplitude, dtype=np.uint8)
    cv2.imshow('frame', result)

cap.release()
cv2.destroyAllWindows()

