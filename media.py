import cv2
import dlib
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull
import numpy as np
from imutils import face_utils

PREDICTOR_PATH = "DLIB/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
Nose_points_2=  list(range(29, 33))

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))


#that method has been just to find out the centre of the eye, using Convex Hull. . .. .

def eye_size(eye):
    eyeWidth = dist.euclidean(eye[0], eye[3])
    hull = ConvexHull(eye)
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)
    eyeCenter = eyeCenter.astype(int)
    return int(eyeWidth), eyeCenter




#That method is for placing the Eye mask on the Eye convex Hull . . .
def place_eye(frame, eyeCenter, eyeSize):
    eyeSize = int(eyeSize * 1.4)
    x1 = int(eyeCenter[0, 0] - (eyeSize / 2))
    x2 = int(eyeCenter[0, 0] + (eyeSize / 2))
    y1 = int(eyeCenter[0, 1] - (eyeSize / 2))
    y2 = int(eyeCenter[0, 1] + (eyeSize / 2))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    #Calculating the size of overlayed image to be placed on the relevant
    eyeOverlayWidth = x2 - x1
    eyeOverlayHeight = y2 - y1

    # calculate the masks for the overlay

    eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)

    mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

imgEye = cv2.imread('Eye.png', -1)

orig_mask = imgEye[:, :, 3]

# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)
# Convert the overlay image image to BGR
# and save the original image size

imgEye = imgEye[:, :, 0:3]

origEyeHeight, origEyeWidth = imgEye.shape[:2]

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 0)
        # dets = DL_detectore(gray,0)
        for det in dets:
            # bbox = det.rect
            bbox= det
            x = bbox.left()
            y = bbox.top()
            x1 = bbox.right()
            y1 = bbox.bottom()

            #that code can be used for the
            shape = predictor(gray, bbox)
            shape = face_utils.shape_to_np(shape)
            (x_new, y_new, w, h) = face_utils.rect_to_bb(bbox)
            cv2.rectangle(frame, (x_new, y_new), (x_new + w, y_new + h), (0, 255, 0), 4)

            for x, y in shape:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, bbox).parts()])

            # print(landmarks)
            left_eye = landmarks[LEFT_EYE_POINTS]
            right_eye = landmarks[RIGHT_EYE_POINTS]

            leftEyeSize, leftEyeCenter = eye_size(left_eye)
            rightEyeSize, rightEyeCenter = eye_size(right_eye)

            place_eye(frame, leftEyeCenter, leftEyeSize)
            place_eye(frame, rightEyeCenter, rightEyeSize)

        cv2.imshow("Faces with Overlay", frame)
        # exit()

    ch = 0xFF & cv2.waitKey(1)

    if ch == ord('q'):
        break
    elif ch==ord('s'):
        cv2.imwrite("Screen_Shot2.png", frame)
        print("Done")
        break


cv2.destroyAllWindows()



