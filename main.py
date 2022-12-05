import os
import cv2 as cv
import numpy as np
import time
from picrawler import Picrawler
crawler = Picrawler([10, 11, 12, 4, 5, 6, 1, 2, 3, 7, 8, 9])                                                            # Instantiate robot [servo numbers]


#Global parameters:
SPEED = 100                                                                                                             # Crawler move speed (%)
# TIME_BETWEEN_ACTIONS = 0.05
FEED_RES_W = 480                                                                                                        # Video feed resolution width
FEED_RES_H = 360                                                                                                        # Video feed resolution height
GRAY_FILTER = 32                                                                                                        # Brightness filter value
ARC_LENGTH_THRESHOLD = 50                                                                                               # Arc length threshold for filtering
TRACE_ROUGHNESS_FACTOR = 0.10                                                                                           # Error factor for polygon tracing
STAGNATION_LIMIT = 1                                                                                                    # Frames to "hold" last detected shape
ASPECT_LOCUS = 0.9                                                                                                      # Aspect ratio fuzziness parameter
ASPECT_FWHM = 0.5                                                                                                       # Aspect ratio fuzziness parameter
SIDES_FWHM = 2.5                                                                                                        # Sides count fuzziness parameter
FILTER_CONF = 0.01
INVEST_CONF = 0.1
GOAL_CONF = 0.2


def look(camIn, prevConfIn, prevRectIn, stagnationIn):
    prevRect = prevRectIn
    prevConf = prevConfIn
    bestRect = prevRect
    bestConf = 0.0
    ret, frame = camIn.read()                                                                                           # frame = source video frame image
    ARframe = frame.copy()                                                                                              # copy frame for final output
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsvFrame = cv.GaussianBlur(hsvFrame, (51, 51), cv.BORDER_DEFAULT)
    lowerHSV1 = np.array([0, 128, 32])                                                                                  # range values specifying HSV masks
    upperHSV1 = np.array([24, 255, 255])
    lowerHSV2 = np.array([164, 128, 32])
    upperHSV2 = np.array([180, 255, 255])
    mask1 = cv.inRange(hsvFrame, lowerHSV1, upperHSV1)                                                                  # HSV mask to cover the bottom hues
    mask2 = cv.inRange(hsvFrame, lowerHSV2, upperHSV2)                                                                  # HSV mask to cover the top hues
    mask = cv.bitwise_or(mask1, mask2)
    masked = cv.bitwise_and(frame, frame, mask=mask)                                                                    # masked frame image
    masked = cv.GaussianBlur(masked, (51, 51), cv.BORDER_DEFAULT)
    boundaries = cv.Canny(masked, 40, 60)                                                                               # edge detection
    contours, hierarchies = cv.findContours(boundaries, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)                           # convert edges to contours
    filteredContours = []
    for contour in contours:
        L = cv.arcLength(contour, True)                                                                                 # filter each contour by arcLength
        if L >= ARC_LENGTH_THRESHOLD:
            newContour = cv.approxPolyDP(contour, int(L * TRACE_ROUGHNESS_FACTOR), True)                                # then simplify curve
            if cv.arcLength(newContour, True) > ARC_LENGTH_THRESHOLD:
                filteredContours.append(newContour)                                                                     # if still long enough: save contour
    contourField = np.zeros(frame.shape, dtype="uint8")                                                                 # new canvas for contours
    cv.drawContours(contourField, filteredContours, -1, (255, 0, 255), 2)
    traceField = np.zeros(frame.shape, dtype="uint8")                                                                   # new canvas for isolated shapes
    nowLocated = 0                                                                                                         # found shapes counter
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
                    nowLocated += 1
                    bestConf = finalConfidence
                    bestRect = (x, y, w, h)
                    cv.putText(traceField, str(L), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)  # put label on shape canvas
    if nowLocated == 0:
        stagnationIn += 1                                                                                               # if no shapes found, increment counter
        if stagnationIn < STAGNATION_LIMIT:
            x, y, w, h = prevRect                                                                                       # if within limit, use last good rect
            f = max(0, 255 - int(stagnationIn * (255 / STAGNATION_LIMIT)))                                              # 'fading' font color for shape canvas
            cv.putText(traceField, "•", (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (f, f, f), 2)               # put label at last good location
            cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                     # redraw last good rectangles on output
            cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)
            cv.putText(ARframe, "goal %.3f" % prevConf, (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 2)          # redraw last good label on output
    else:
        stagnationIn = 0                                                                                                # if >0 shapes found, reset counter
        x, y, w, h = bestRect
        prevRect = bestRect
        prevConf = bestConf
        cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                         # draw bounding rectangle on output
        cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)                       # draw 'name tab' rectangle on output
        cv.putText(ARframe, "goal %.3f" % bestConf, (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.0, (0, 0, 0), 2)                                                                                   # put label on "tab" in output

    # cv.imshow("mask1", mask1)                                                                                           # DEBUG: lower HSV mask
    # cv.imshow("mask2", mask2)                                                                                           # DEBUG: upper HSV mask
    # cv.imshow("OR mask", mask)                                                                                          # DEBUG: combined HSV mask
    # cv.imshow("masked", masked)                                                                                         # DEBUG: HSV masked output
    # cv.imshow("threshmasked", threshmasked)                                                                             # DEBUG: HSV/threshold masked output
    # cv.imshow("Canny", boundaries)                                                                                      # DEBUG: edge detection output
    # cv.imshow("contours", contourField)                                                                                 # DEBUG: contour output
    # cv.imshow("shape traces", traceField)                                                                               # DEBUG: traced shapes output
    # cv.imshow("goal locator", ARframe)                                                                                  # final "augmented reality" output
    if prevConf > GOAL_CONF:
        cv.imwrite("foundGoal.png", ARframe)
        print("goal image saved to: " + str(os.getcwd()))
    return nowLocated, prevConf, prevRect, stagnationIn


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
    elif n == 3 or n == 4:
        return 1.0
    else:
        return gaussian(n, 4, SIDES_FWHM, 'z')


def fuzzifyColor(colorListIn):
    b = colorListIn[0]
    bConf = gaussian(b, 64, 128, 'z')
    g = colorListIn[1]
    gConf = gaussian(g, 64, 128, 'z')
    r = colorListIn[2]
    rConf = gaussian(r, 192, 128, 's')
    finalConf = bConf * gConf * rConf
    # print("colorConf: %.2f" % bConf + " %.2f" % gConf + " %.2f" % rConf + " = %.2f" % finalConf)
    return finalConf


if __name__ == "__main__":
    cam = cv.VideoCapture(0)                                                                                            # instantiate cam feed
    cam.set(3, FEED_RES_W)                                                                                              # set feed resolution
    cam.set(4, FEED_RES_H)
    previousConf = 0.0
    previousRect = (0, 0, 0, 0)                                                                                         # storage to remember rectangle
    stagnation = 0                                                                                                      # counter to hold "lost" shapes
    lookStep = -1
    lookCycle = ["turn left", "turn left", "turn right", "turn right", "turn right", "turn right", "turn left", "turn left", "forward", "forward"]

    loop = True                                                                                                         # boolean loop variable
    while loop:
        prevStagnation = stagnation
        #nowLocated, prevConf, prevRect, stagnationIn                                                                   # reminder of outputs
        located, previousConf, previousRect, stagnation = look(cam, previousConf, previousRect, stagnation)
        time.sleep(1)
        if located > 0 and previousConf > INVEST_CONF:
            if previousConf > GOAL_CONF:
                print("GOAL FOUND!!!\nconfidence = %.3f" % previousConf)
                crawler.do_action('dance', 1, SPEED)
                loop = False
            else:
                print("Possible goal found! Investigating...\nconfidence = %.3f" % previousConf)
                crawler.do_action("forward", 2, SPEED)
                time.sleep(0.01)
        else:
            lookStep = (lookStep + 1) % len(lookCycle)
            print("Goal not found...\nMovement step %d" % lookStep + ": " + lookCycle[lookStep])
            crawler.do_action(lookCycle[lookStep], 1, SPEED)
            time.sleep(0.01)

        if cv.waitKey(1) & 0xFF == ord('q'):
            loop = False

    cam.release()                                                                                                       # safely stop video feed
    cv.destroyAllWindows()                                                                                              # close all opened windows