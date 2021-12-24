import cv2
import numpy as np


def dualPCA(data):

    mean = np.zeros((8000))
    len = data.shape[1]

    for i in range(8000):
        mean[i] = np.sum(data[i]) / len
        data[i] -= mean[i]

    C = (1/7999) * data.T.dot(data)
    U, S, _ = np.linalg.svd(C)
    U = data.dot(U)*(1/(S*7999))**0.5

    return U, mean

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Space - to take sample image")  # 32
print("Enter - to run face recognition")  # 13
print("A, S, D, W - move detection rectangle")  # 13

sampling = True
recognition = False
outline = np.array([150, 150, 150])
data_set = []
matrix = None
mean = None
threshold = 0

x1 = 140
x2 = 340
y1 = 240
y2 = 400

ret, frame = cap.read()
frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("Face recognition", frame)

while  cv2.getWindowProperty("Face recognition", cv2.WND_PROP_VISIBLE) == 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame[x1-1:x1, y1:y2] = outline
    frame[x2:x2+1, y1:y2] = outline
    frame[x1:x2, y1-1:y1] = outline
    frame[x1:x2, y2:y2+1] = outline
    cv2.imshow("Face recognition", frame)

    input = cv2.waitKey(1)
    if input > 0:

        if (input == 32 and sampling):
            gray = cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY)
            cv2.imshow("Display", gray)
            gray = cv2.resize(gray, (80, 100))
            data_set.append(np.reshape(gray, 8000).astype(np.float32))

        elif (input == 13):

            recognition = not recognition

            if sampling:
                sampling = False
                data_set = np.array(data_set)
                matrix, mean = dualPCA(data_set.T)

                for img in data_set:
                    imag = np.copy(img)
                    imag = imag - np.amin(imag)
                    imag = imag / np.amax(imag) * 255

                    proj = matrix.T.dot(imag - mean)
                    proj = matrix.dot(proj) + mean
                    proj = proj - np.amin(proj)
                    proj = proj / np.amax(proj) * 255
                    threshold += ( 0.5 * np.sum( ( imag**0.5 - proj**0.5 )**2 ) )**0.5
                threshold = threshold / data_set.shape[0] * 1.25

        elif (input == 97):
            if y1 >= 4:
                y1 -= 4
                y2 -= 4

        elif (input == 100):
            if y2 < 636:
                y1 += 4
                y2 += 4

        elif (input == 119):
            if x1 >= 4:
                x1 -= 4
                x2 -= 4

        elif (input == 115):
            if x2 < 476:
                x1 += 4
                x2 += 4

    if recognition:

        gray = cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (80, 100))

        proj = np.reshape(gray, 8000)
        proj = matrix.T.dot(proj - mean)
        proj = matrix.dot(proj) + mean
        proj = np.reshape(proj, (100, 80))

        proj = proj - np.amin(proj)
        proj = proj / np.amax(proj) * 255
        dist = ( 0.5 * np.sum( ( gray**0.5 - proj**0.5 )**2 ) )**0.5
        print(f"Euclidian distance: {dist} thr: {threshold}")

        proj = cv2.resize(proj, (160, 200))
        cv2.imshow("Display", proj.astype(np.uint8))

        if dist < threshold:
            outline = outline = np.array([0, 200, 0])
        else:
            outline = outline = np.array([0, 0, 200])

cap.release()
cv2.destroyAllWindows()
