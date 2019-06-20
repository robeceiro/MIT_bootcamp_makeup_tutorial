# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import RPi.GPIO as GPIO

lightValue = 0

def button_callback1(channel):
        print("Button 1 was pushed!")
def button_callback2(channel):
        print("Button 2 was pushed!")
def button_callback3(channel):
        print("Button 3 was pushed!")
def button_callback4(channel):
        if lightValue < 3:
            lightValue = lightValue + 1
        else:
            lightValue=0
        print("Button 4 was pushed!")
        print(str(lightValue))


GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD)
#GPIO.setmode(GPIO.BCM)
#Light output
GPIO.setup(36, GPIO.OUT)
GPIO.setup(38, GPIO.OUT)
GPIO.setup(40, GPIO.OUT)
GPIO.output(36, GPIO.LOW)
GPIO.output(38, GPIO.LOW)
GPIO.output(40, GPIO.LOW)
#Button Input
GPIO.setup(8, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 8 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 12 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 16 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(8,GPIO.RISING,callback=button_callback4,bouncetime=200) # Setup event on pin 8 rising edge
GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback3,bouncetime=200) # Setup event on pin 10 rising edge
GPIO.add_event_detect(12,GPIO.RISING,callback=button_callback2,bouncetime=200) # Setup event on pin 12 rising edge
GPIO.add_event_detect(16,GPIO.RISING,callback=button_callback1,bouncetime=200) # Setup event on pin 16 rising edge



# construct the argument parser and parse the arguments
'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

#picamera : An optional command line argument, this switch indicates whether the Raspberry Pi camera module should be used instead of the default webcam/USB camera. Supply a value > 0 to use your Raspberry Pi camera.
'''
args = {}
args["shape_predictor"] = "shape_predictor_68_face_landmarks.dat"
args["picamera"] = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.8):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    height,width,channels = image.shape
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
            (168, 100, 168), (158, 163, 32),
            (163, 38, 32), (180, 42, 220)]
    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "right_eyebrow": # or name == "left_eyebrow" or name=="mouth":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            text = 'Step 1: Paint here'
            font = cv2.FONT_HERSHEY_SIMPLEX
            radius = 20
            cv2.putText(overlay, text, (2*radius+5, height-10), font, 0.5, (255, 255, 0), 2)
            cv2.circle(overlay,(radius,height-radius), radius, (0,0,255), -1)
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        #else:
        #    hull = cv2.convexHull(pts)
        #    cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # return the output image
    return output

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
     # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        output = visualize_facial_landmarks(frame, shape)

    # show the frame
    cv2.imshow("Frame", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
