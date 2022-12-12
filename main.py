import threading                                                                                                        # import threading for cam
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
LOWER_HSV1 = np.array([0, 128, 32])                                                                                     # Lower bounds for bottom-end HSV mask
UPPER_HSV1 = np.array([8, 255, 255])                                                                                    # Upper bounds for bottom-end HSV mask
LOWER_HSV2 = np.array([164, 96, 32])                                                                                    # Lower bounds for top-end HSV mask
UPPER_HSV2 = np.array([180, 255, 255])                                                                                  # Upper bounds for top-end HSV mask
ARC_LENGTH_THRESHOLD = 120                                                                                              # Arc length threshold for filtering
TRACE_ROUGHNESS = 5
ASPECT_LOCUS = 1.0                                                                                                      # Aspect ratio fuzziness center
ASPECT_FWHM = 0.5                                                                                                       # Aspect ratio fuzziness width
SIDES_LOCUS = 4.0                                                                                                       # Sides count fuzziness center
SIDES_FWHM = 2.5                                                                                                        # Sides count fuzziness width
COLOR_LOCUS = 0.0                                                                                                       # Color ratio fuzziness center
COLOR_FWHM = 1.33                                                                                                       # Color ratio fuzziness width
LOOK_FRAMES = 5
HOLD_TIME = 0.1                                                                                                         # delay time between ops
FILTER_CONF = 0.01                                                                                                      # absolute minimum conf for recognition
INVEST_CONF = 0.1                                                                                                       # conf to trigger investigation
GOAL_CONF = 0.5                                                                                                         # conf to declare goal found
CONFIDENCE = 0.0                                                                                                        # global confidence variable


def look(camIn):
    bestConf = 0.0                                                                                                      # best confidence container
    ret, frame = camIn.read()                                                                                           # frame = source video frame image
    ARframe = frame.copy()                                                                                              # copy frame for final output
    hsvFrame = cv.GaussianBlur(frame, (9, 9), cv.BORDER_DEFAULT)                                                        # blur frame copy
    hsvFrame = cv.cvtColor(hsvFrame, cv.COLOR_BGR2HSV)                                                                  # convert copy to HSV
    mask1 = cv.inRange(hsvFrame, LOWER_HSV1, UPPER_HSV1)                                                                # HSV mask to cover the bottom hues
    mask2 = cv.inRange(hsvFrame, LOWER_HSV2, UPPER_HSV2)                                                                # HSV mask to cover the top hues
    mask = cv.bitwise_or(mask1, mask2)                                                                                  # combine masks
    masked = cv.bitwise_and(frame, frame, mask=mask)                                                                    # masked frame image
    masked = cv.GaussianBlur(masked, (9, 9), cv.BORDER_DEFAULT)                                                         # blur new working image
    boundaries = cv.Canny(masked, 40, 60)                                                                               # edge detection
    contours, hierarchies = cv.findContours(boundaries, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)                           # convert edges to contours
    filteredContours = []                                                                                               # instantiate empty list
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
        if len(contour) > 1:                                                                                            # filter out non-shapes
            sidesConf = fuzzifyNumSides(len(contour))                                                                   # fuzzify number of sides
            cv.drawContours(traceField, [contour], 0, (255, 255, 255), 3)                                               # draw shape to shape canvas
            x, y, w, h, = cv.boundingRect(contour)                                                                      # get rectangle parameters
            A = float(w)/float(h)                                                                                       # get aspect ratio
            aspectConf = fuzzifyApsect(A)                                                                               # fuzzify aspect ratio
            iniConfidence = min(sidesConf, aspectConf)                                                                  # judge initial confidence
            if iniConfidence >= FILTER_CONF:                                                                            # filter further by ini. conf.
                prospect = frame[y:y+h, x:x+w]                                                                          # isolate region of interest
                avgColor = (np.average(prospect[:, :, 0]), np.average(prospect[:, :, 1]), np.average(prospect[:, :, 2]))  # average color of crop
                colorConf = fuzzifyColor(avgColor)                                                                      # fuzzify average color
                finalConfidence = min(iniConfidence, colorConf)                                                         # judge final confidence
                if finalConfidence > INVEST_CONF and finalConfidence > bestConf:                                        # proceed only if best so far
                    L = cv.arcLength(contour, True)                                                                     # number of sides
                    bestConf = finalConfidence                                                                          # save confidence
                    bestRect = (x, y, w, h)                                                                             # save region info
                    cv.putText(traceField, str(L), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)  # put label on shape canvas
    if bestConf > INVEST_CONF:
        x, y, w, h = bestRect
        cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                         # draw bounding rectangle on output
        cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)                       # draw 'name tab' rectangle on output
        cv.putText(ARframe, "goal %.3f" % bestConf, (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.0, (0, 0, 0), 2)                                                                                   # put label on "tab" in output

    if CAMERA_TEST:
        # Uncomment lines to see specific intermediate processing steps:
        # cv.imshow("mask1", mask1)                                                                                       # DEBUG: lower HSV mask
        # cv.imshow("mask2", mask2)                                                                                       # DEBUG: upper HSV mask
        # cv.imshow("OR mask", mask)                                                                                      # DEBUG: combined HSV mask
        # cv.imshow("masked", masked)                                                                                     # DEBUG: HSV masked output
        # cv.imshow("Canny", boundaries)                                                                                  # DEBUG: edge detection output
        # cv.imshow("contours", contourField)                                                                             # DEBUG: contour output
        # cv.imshow("shape traces", traceField)                                                                           # DEBUG: traced shapes output
        cv.imshow("goal locator", ARframe)                                                                              # final "augmented reality" output
        pass                                                                                                            # in case all are disabled

    # cv.imshow("goal locator", ARframe)                                                                                  # DEBUG: video output every call

    if (not CAMERA_TEST) and (bestConf > GOAL_CONF):
        cv.imwrite("foundGoal.png", ARframe)
    global CONFIDENCE
    CONFIDENCE = bestConf


def gaussian(xIn, locusIn, widthIn, shapeIn='g', maxIn=1.0):                                                            # Gaussian func for fuzzifying
    # Gaussian parameter variables:
    a = float(maxIn)
    b = float(locusIn)
    w = float(widthIn)
    x = float(xIn)
    result = 0.0
    # if using s- or z-curve, first check to see if in maxed region:
    if (shapeIn == 's' and x >= b) or (shapeIn == 'z' and x <= b):
        result = 1.0
    else:
        result = a * np.exp(-4.0 * np.log(2) * ((x - b) * (x - b)) / (w * w))                                           # gaussian function in terms of FWHM
        #SOURCE: https://en.wikipedia.org/wiki/Gaussian_function#Properties "in terms of the FWHM, represented by w"
    return result


def fuzzifyApsect(aspectIn):                                                                                            # goal confidence from aspect ratio
    A = float(aspectIn)
    if A > 1.0:                                                                                                         # if A > 1, normalize
        A = 1.0 / A                                                                                                     # (use inverse)
    return gaussian(A, ASPECT_LOCUS, ASPECT_FWHM, 's')


def fuzzifyNumSides(numIn):
    n = int(round(numIn))
    if n < 2:                                                                                                           # 1 or 0 sides = disqualify
        return 0.0
    elif n == 2:                                                                                                        # 2 sides is kinda sus
        return 0.5
    elif 3 <= n <= 5:                                                                                                   # fine if within 3-5
        return 1.0
    else:                                                                                                               # if greater: Gaussian falloff
        return gaussian(n, SIDES_LOCUS, SIDES_FWHM, 'z')


def fuzzifyColor(colorListIn):                                                                                          # fuzzied var = ratio of avg(G,B) to R:
    r = float(colorListIn[2])
    g = float(colorListIn[1])
    b = float(colorListIn[0])
    gb = (g + b) / 2.0
    ratio = gb / r
    return gaussian(ratio, COLOR_LOCUS, COLOR_FWHM, 'z')                                                                # Gaussian: lower ratio = higher conf.


def doMove(moveName, distance):                                                                                         # perform named move
    if moveName is not None:
        if moveName == "forward" and (0 <= distance < 10):                                                              # if forward, check there's room:
            print("ERROR: Can't move! No space!!!")
        else:
            crawler.do_action(moveName, 1, SPEED)


if __name__ == "__main__":                                                                                              # do not run on file import
    cam = cv.VideoCapture(0)                                                                                            # instantiate cam feed
    cam.set(3, FEED_RES_W)                                                                                              # set feed resolution
    cam.set(4, FEED_RES_H)
    camThread = None                                                                                                    # declare camera thread
    frontDistance = 0.0
    moveStep = -1                                                                                                       # dictates which movement to make
    moveCycle = ["turn left", "turn left", "turn left",
                 "turn right", "turn right", "turn right",
                 "turn right", "turn right", "turn right",
                 "turn left", "turn left", "turn left",
                 "forward", "forward", "forward"]                                                                       # "search pattern" sequence

    loopCount = 0
    loop = True                                                                                                         # boolean loop variable
    while loop:
        maxConf = 0.0
        minDist = 255
        for i in range(0, LOOK_FRAMES):
            camThread = threading.Thread(target=look, args=(cam,))
            camThread.start()                                                                                           # start camThread
            frontDistance = sonar.read()
            if (camThread is not None) and (camThread.isAlive()):                                                       # wait until camThread finishes
                camThread.join()
            if CONFIDENCE > maxConf:
                maxConf = CONFIDENCE
            if frontDistance < minDist:
                minDist = frontDistance

        if not CAMERA_TEST:                                                                                             # only move if NOT testing camera
            if maxConf > INVEST_CONF:                                                                                   # if SOME confidence...
                if maxConf > GOAL_CONF:                                                                                 # if ALOT of confidence...
                    print("GOAL FOUND!!!\nconfidence = %.3f" % maxConf)                                                 # shout, sit, win
                    doMove("sit", minDist)
                    loop = False
                else:                                                                                                   # if POSSIBLE goal...
                    if minDist >= 10 or minDist < 0:                                                                    # ...and room in front, move forward
                        print("Possible goal found! Investigating...\nconfidence = %.3f" % maxConf)
                        doMove("forward", minDist)
                    else:                                                                                               # otherwise, stay and look some more.
                        print("Possible goal found! Waiting...\nconfidence = %.3f" % maxConf)
            else:
                moveStep = (moveStep + 1) % len(moveCycle)                                                              # iterate through moveCycle
                while moveCycle[moveStep] == "forward" and 0 <= minDist < 10:
                    moveStep = (moveStep + 1) % len(moveCycle)
                print("Goal not found...\nMovement step %d" % moveStep + ": " + moveCycle[moveStep])
                doMove(moveCycle[moveStep], minDist)

        if cv.waitKey(1) & 0xFF == ord('q'):
            loop = False

    cam.release()                                                                                                       # safely stop video feed
    cv.destroyAllWindows()                                                                                              # close all opened windows