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

global current_step
current_step = 1
global light
light = 0
global snapshot
snapshot=False

def update_light():
        global light
        if light==0:
             GPIO.output(36, GPIO.LOW)
             GPIO.output(38, GPIO.LOW)
             GPIO.output(40, GPIO.LOW)
        elif light==1:
             GPIO.output(36, GPIO.HIGH)
             GPIO.output(38, GPIO.LOW)
             GPIO.output(40, GPIO.LOW)
        elif light==2:
             GPIO.output(36, GPIO.LOW)
             GPIO.output(38, GPIO.HIGH)
             GPIO.output(40, GPIO.LOW)
        elif light==3:
             GPIO.output(36, GPIO.LOW)
             GPIO.output(38, GPIO.LOW)
             GPIO.output(40, GPIO.HIGH)


def button_callback1(channel):
        print("Button 1 was pushed!")
        global light
        if light < 3:
            light = light + 1
            update_light()
        else:
            light = 0
            update_light()
def button_callback2(channel):
        print("Button 2 was pushed!")
        global snapshot
        if snapshot:
            snapshot = False
        else:
            snapshot = True
def button_callback3(channel):
        print("Button 3 was pushed!")
        global current_step
        if current_step > 1:
            current_step = current_step - 1
        else:
            current_step = 7
def button_callback4(channel):
        global current_step
        if current_step < 7:
            current_step = current_step + 1
        else:
            current_step=1
        print("Button 4 was pushed!")
        print(str(current_step))


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



def makeup_step(new_name,color, image, shape, fill_shape, step_number, text,alpha=0.8):
    overlay = image.copy()
    output = image.copy()
    radius = 20
    height,width,channels = image.shape
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        if name == new_name:
            if fill_shape:
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, color, -1)
            else:
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    cv2.line(overlay, ptA, ptB, color, 2)
    cv2.rectangle(overlay,(2*radius,5),(250,45),(0,0,0),-1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(output,(radius,20), radius, color, -1)
    cv2.putText(output, text, (2*radius+5, 22), font, 0.5, (255, 255, 255), 1)
    # return the output image
    return output
    

from collections import OrderedDict
STEPS = OrderedDict([
    (3,"mouth"),
    (1,"right_eyebrow"),
    (2,"left_eyebrow"),
    (6,"right_eye"),
    (5,"left_eye"),
    (4,"nose"),
    (7,"jaw")
])
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
default_color = (0,170,220)
COLORS = OrderedDict([
    ("mouth", (0, 200, 0)),
    ("right_eyebrow", default_color),
    ("left_eyebrow", default_color),
    ("right_eye", default_color),
    ("left_eye", default_color),
    ("nose", default_color),
    ("jaw", default_color)
])
TEXTS = OrderedDict([
    ("mouth", "L'Oreal Lipstick Mate N.68"),
    ("right_eyebrow", "L'Oreal Eyeliner N.51"),
    ("left_eyebrow", "L'Oreal Eyeliner N.51"),
    ("right_eye", "L'Oreal Eyeliner N.51"),
    ("left_eye", "L'Oreal Eyeliner N.51"),
    ("nose", "L'Oreal Paris Infalible"),
    ("jaw", "Powder")
])

from InstagramAPI import InstagramAPI

instagram = InstagramAPI("suyuen","makeuplovers")
instagram.login()
light_enable = False
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    rects = detector(gray, 0)
    if rects and not light_enable:
        GPIO.output(36, GPIO.HIGH)
        GPIO.output(38, GPIO.LOW)
        GPIO.output(40, GPIO.LOW)
        light_enable = True
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #output = visualize_facial_landmarks(frame, shape)
        output = makeup_step(STEPS[current_step],COLORS[STEPS[current_step]], frame, shape, True,current_step,TEXTS[STEPS[current_step]],0.4)
    # global snapshot
    if not snapshot:
        cv2.imshow("Frame", output)
    else:
        path = "upload.jpg"
        cv2.imwrite(path,output)
        output = cv2.putText(output,"Uploading to Instagram",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) 
        cv2.imshow("Frame",output)
        caption = "#mitbootcamp #mitdeeptech MakeUpLovers"
        instagram.uploadPhoto(path,caption=caption)
        snapshot = False
        #time.sleep(6)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
