import time                                                                                                             # import time for sleep
import cv2 as cv                                                                                                        # import openCV-contrib-python
import numpy as np                                                                                                      # import numpy
from picrawler import Picrawler                                                                                         # import mfg. fcns for robot
from robot_hat import Ultrasonic                                                                                        # import mfg  fcns for sonar
from robot_hat import Pin                                                                                               # import dependencies for mfg. fcns
crawler = Picrawler([10, 11, 12, 4, 5, 6, 1, 2, 3, 7, 8, 9])                                                            # Instantiate robot [servo numbers]
sonar = Ultrasonic(Pin("D2"), Pin("D3"))                                                                                # Instantiate sonar sensor


#Global parameters:
CAMERA_TEST = False                                                                                                     # Toggles "cam test mode"--no movement
SPEED = 100                                                                                                             # Crawler move speed (%)
FEED_RES_W = 480                                                                                                        # Video feed resolution width
FEED_RES_H = 360                                                                                                        # Video feed resolution height
LOWER_HSV1 = np.array([0, 128, 32])                                                                                     # lowerbounds for low-end HSV mask
UPPER_HSV1 = np.array([8, 255, 255])                                                                                    # upperbounds for low-end HSV mask
LOWER_HSV2 = np.array([164, 96, 32])                                                                                    # lowerbounds for high-end HSV mask
UPPER_HSV2 = np.array([180, 255, 255])                                                                                  # upperbounds for high-end HSV mask
ARC_LENGTH_THRESHOLD = 120                                                                                               # Arc length threshold for filtering
TRACE_ROUGHNESS = 5
ASPECT_LOCUS = 1.0                                                                                                      # Aspect ratio fuzziness center
ASPECT_FWHM = 0.5                                                                                                       # Aspect ratio fuzziness width
SIDES_LOCUS = 4.0                                                                                                       # Sides count fuzziness center
SIDES_FWHM = 2.5                                                                                                        # Sides count fuzziness width
COLOR_LOCUS = 0.0                                                                                                       # Color ratio fuzziness center
COLOR_FWHM = 1.33                                                                                                       # Color ratio fuzziness width
WAIT_FRAMES = 20
MOVE_FRAMES = 25                                                                                                         # moves execute once per this many loops
HOLD_TIME = 0.25                                                                                                        # delay time between ops
FILTER_CONF = 0.01                                                                                                      # absolute minimum conf for recognition
INVEST_CONF = 0.1                                                                                                       # conf to trigger investigation
GOAL_CONF = 0.5                                                                                                         # conf to declare goal found


def look(camIn):
    bestRect = (0, 0, 0, 0)
    bestConf = 0.0
    ret, frame = camIn.read()                                                                                           # frame = source video frame image
    ARframe = frame.copy()                                                                                              # copy frame for final output
    hsvFrame = cv.GaussianBlur(frame, (9, 9), cv.BORDER_DEFAULT)
    hsvFrame = cv.cvtColor(hsvFrame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsvFrame, LOWER_HSV1, UPPER_HSV1)                                                                # HSV mask to cover the bottom hues
    mask2 = cv.inRange(hsvFrame, LOWER_HSV2, UPPER_HSV2)                                                                # HSV mask to cover the top hues
    mask = cv.bitwise_or(mask1, mask2)
    masked = cv.bitwise_and(frame, frame, mask=mask)                                                                    # masked frame image
    masked = cv.GaussianBlur(masked, (9, 9), cv.BORDER_DEFAULT)
    boundaries = cv.Canny(masked, 40, 60)                                                                               # edge detection
    contours, hierarchies = cv.findContours(boundaries, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)                           # convert edges to contours
    filteredContours = []
    for contour in contours:
        L = cv.arcLength(contour, True)                                                                                 # filter each contour by arcLength
        if L >= ARC_LENGTH_THRESHOLD:
            newContour = cv.approxPolyDP(contour, int(TRACE_ROUGHNESS), True)                                           # then simplify curve
            if cv.arcLength(newContour, True) > ARC_LENGTH_THRESHOLD:
                filteredContours.append(newContour)                                                                     # if still long enough: save contour
    contourField = np.zeros(frame.shape, dtype="uint8")                                                                 # new canvas for contours
    cv.drawContours(contourField, filteredContours, -1, (255, 0, 255), 2)
    traceField = np.zeros(frame.shape, dtype="uint8")                                                                   # new canvas for isolated shapes
    for contour in filteredContours:
        if len(contour) > 1:
            sidesConf = fuzzifyNumSides(len(contour))
            cv.drawContours(traceField, [contour], 0, (255, 255, 255), 3)                                               # draw shape to shape canvas
            x, y, w, h, = cv.boundingRect(contour)                                                                      # get rectangle parameters
            A = float(w)/float(h)
            aspectConf = fuzzifyApsect(A)
            iniConfidence = sidesConf * aspectConf
            if iniConfidence >= FILTER_CONF:
                prospect = frame[y:y+h, x:x+w]
                avgColor = (np.average(prospect[:, :, 0]), np.average(prospect[:, :, 1]), np.average(prospect[:, :, 2]))
                colorConf = fuzzifyColor(avgColor)
                finalConfidence = iniConfidence * colorConf
                if finalConfidence > INVEST_CONF and finalConfidence > bestConf:
                    L = cv.arcLength(contour, True)
                    bestConf = finalConfidence
                    bestRect = (x, y, w, h)
                    cv.putText(traceField, str(L), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)  # put label on shape canvas
                # if finalConfidence > GOAL_CONF:
                #     print("FOUND!!! sides:%.3f" % sidesConf + "\taspect:%.3f" % aspectConf + "\tcolor:%.3f" % colorConf + "\tfinal:%.3f" % finalConfidence)
                # else:
                #     print("found... sides:%.3f" % sidesConf + "\taspect:%.3f" % aspectConf + "\tcolor:%.3f" % colorConf + "\tfinal:%.3f" % finalConfidence)
    if bestConf > INVEST_CONF:
        x, y, w, h = bestRect
        cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                         # draw bounding rectangle on output
        cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)                       # draw 'name tab' rectangle on output
        cv.putText(ARframe, "goal %.3f" % bestConf, (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.0, (0, 0, 0), 2)                                                                                   # put label on "tab" in output

    if CAMERA_TEST:
        # cv.imshow("mask1", mask1)                                                                                       # DEBUG: lower HSV mask
        # cv.imshow("mask2", mask2)                                                                                       # DEBUG: upper HSV mask
        # cv.imshow("OR mask", mask)                                                                                      # DEBUG: combined HSV mask
        # cv.imshow("masked", masked)                                                                                     # DEBUG: HSV masked output
        # cv.imshow("Canny", boundaries)                                                                                  # DEBUG: edge detection output
        # cv.imshow("contours", contourField)                                                                             # DEBUG: contour output
        # cv.imshow("shape traces", traceField)                                                                           # DEBUG: traced shapes output
        cv.imshow("goal locator", ARframe)                                                                              # final "augmented reality" output
        pass                                                                                                            # in case all disabled
    cv.imshow("goal locator", ARframe)
    if (not CAMERA_TEST) and (bestConf > GOAL_CONF):
        cv.imwrite("foundGoal.png", ARframe)
    return bestConf, bestRect


def gaussian(xIn, locusIn, widthIn, shapeIn='g', maxIn=1.0):
    a = float(maxIn)
    b = float(locusIn)
    w = float(widthIn)
    x = float(xIn)
    result = 0.0
    if (shapeIn == 's' and x >= b) or (shapeIn == 'z' and x <= b):                                                      # if s- or z-curve, first check if max
        result = 1.0
    else:
        result = a * np.exp(-4.0 * np.log(2) * ((x - b) * (x - b)) / (w * w))                                           # gaussian function in terms of FWHM
        #SOURCE: https://en.wikipedia.org/wiki/Gaussian_function#Properties "in terms of the FWHM, represented by w"
    return result


def fuzzifyApsect(aspectIn):                                                                                            # goal confidence from aspect ratio
    A = float(aspectIn)
    if A > 1.0:
        A = 1.0 / A
    return gaussian(A, ASPECT_LOCUS, ASPECT_FWHM, 's')


def fuzzifyNumSides(numIn):
    n = int(round(numIn))
    if n < 2:
        return 0.0
    elif n == 2:
        return 0.5
    elif n == 3 or n == 4 or n == 5:
        return 1.0
    else:
        return gaussian(n, SIDES_LOCUS, SIDES_FWHM, 'z')


def fuzzifyColor(colorListIn):                                                                                          # average g and b; check ratio to r
    r = float(colorListIn[2])
    g = float(colorListIn[1])
    b = float(colorListIn[0])
    # print("colors: %d" % r + ", %d" % g + ", %d" % b)
    gb = (g + b) / 2.0
    ratio = gb / r
    # print("color ratio: %.3f" % ratio)
    colorConf = gaussian(ratio, COLOR_LOCUS, COLOR_FWHM, 'z')
    return colorConf


if __name__ == "__main__":
    cam = cv.VideoCapture(0)                                                                                            # instantiate cam feed
    cam.set(3, FEED_RES_W)                                                                                              # set feed resolution
    cam.set(4, FEED_RES_H)
    confidence = 0.0
    waitFrames = -1                                                                                                     # dictates when to wait
    moveFrames = -1                                                                                                     # dictates when to look vs move
    moveStep = -1                                                                                                       # dictates which movement to make
    moveCycle = ["turn left", "turn left", "turn left",
                 "turn right", "turn right", "turn right",
                 "turn right", "turn right", "turn right",
                 "turn left", "turn left", "turn left",
                 "forward", "forward", "forward"]                                                                       # "search pattern" sequence

    loop = True                                                                                                         # boolean loop variable
    while loop:
        frontDistance = sonar.read()                                                                                    # measure distance in front of robot
        # confidence, rectangle = look(cam)                                                                               # get goal confidence and location
        # moveFrames = (moveFrames + 1) % MOVE_FRAMES                                                                     # look for X cycles; move for 1
        # if (not CAMERA_TEST) and ((confidence > GOAL_CONF) or (moveFrames == 0)):                                       # only move if NOT testing camera
        waitFrames = (waitFrames + 1) % WAIT_FRAMES
        if (not CAMERA_TEST) and waitFrames == 0:                                                                                       # only move if NOT testing camera
            if confidence > INVEST_CONF:                                                                                # if SOME confidence...
                if confidence > GOAL_CONF:                                                                              # if ALOT of confidence: shout, sit, win
                    print("GOAL FOUND!!!\nconfidence = %.3f" % confidence)
                    # crawler.do_action('dance', 1, SPEED)      # DO NOT UNCOMMENT! DANGEROUS!
                    crawler.do_action('sit', 1, SPEED)
                    loop = False
                else:                                                                                                   # if POSSIBLE goal...
                    if frontDistance >= 10 or frontDistance < 0:                                                        # ...and room in front, move forward
                        print("Possible goal found! Investigating...\nconfidence = %.3f" % confidence)
                        crawler.do_action("forward", 1, SPEED)
                    else:                                                                                               # otherwise, stay and look some more.
                        print("Possible goal found! Waiting...\nconfidence = %.3f" % confidence)
                        moveStep = -1                                                                                   # reset move cycle to initial position
                    time.sleep(HOLD_TIME)
            else:
                moveStep = (moveStep + 1) % len(moveCycle)                                                              # iterate through moveCycle
                while moveCycle[moveStep] == "forward" and 0 <= frontDistance < 10:
                    moveStep = (moveStep + 1) % len(moveCycle)
                print("Goal not found...\nMovement step %d" % moveStep + ": " + moveCycle[moveStep])
                crawler.do_action(moveCycle[moveStep], 1, SPEED)
                # time.sleep(HOLD_TIME)

            time.sleep(HOLD_TIME)                                                                                       # keep from moving faster than servos
            confidence, rectangle = look(cam)

        if cv.waitKey(1) & 0xFF == ord('q'):
            loop = False

    cam.release()                                                                                                       # safely stop video feed
    cv.destroyAllWindows()                                                                                              # close all opened windows