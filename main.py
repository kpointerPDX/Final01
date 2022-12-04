# import sys
# sys.path.append("/opt/ezblock")
import cv2 as cv
import numpy as np
import time
# from picrawler import Picrawler
# crawler = Picrawler([10, 11, 12, 4, 5, 6, 1, 2, 3, 7, 8, 9])                                                            # Instantiate robot [servo numbers]


#Global parameters:
SPEED = 100
TIME_BETWEEN_ACTIONS = 0.05
FEED_RES_W = 480                                                                                                        # Video feed resolution width
FEED_RES_H = 360                                                                                                        # Video feed resolution height
BGR_FILTER = 64
GRAY_FILTER = 32
ARC_LENGTH_THRESHOLD = 50                                                                                              # Arc length threshold for filtering
# TRACE_ROUGHNESS = 20                                                                                                    # Allowed error from curve for polygon
TRACE_ROUGHNESS_FACTOR = 0.10                                                                                           # Allowed error from curve for polygon
STAGNATION_LIMIT = 1                                                                                                    # Frames to "hold" last detected shape
ASPECT_LOCUS = 0.9
ASPECT_FWHM = 0.5


def showFeed(camIn, prevRectIn, stagnationIn):
    prevRect = prevRectIn
    ret, frame = camIn.read()                                                                                           # frame = source video frame image
    # lowerBGR = np.array([0, 0, 0])                                                                                      # Lowerbound of BGR mask
    # upperBGR = np.array([BGR_FILTER, BGR_FILTER, 255])                                                                  # Upperbound of BGR mask
    # mask = cv.inRange(frame, lowerBGR, upperBGR)                                                                        # BGR mask to filter out non-red pixels
    lowerHSV = np.array([-44, 16, 0])
    upperHSV = np.array([44, 255, 255])
    mask = cv.inRange(frame, lowerHSV, upperHSV)
    masked = cv.bitwise_and(frame, frame, mask=mask)                                                                    # masked frame image
    gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)                                                                       # converted to grayscale
    threshold, thresh = cv.threshold(gray, GRAY_FILTER, 255, cv.THRESH_BINARY)                                          # brightness threshold to create mask
    threshmasked = cv.bitwise_and(masked, masked, mask=thresh)                                                          # masked again by threshold
    gray2 = cv.cvtColor(threshmasked, cv.COLOR_BGR2GRAY)                                                                # threshmasked converted to grayscale
    blur = cv.GaussianBlur(gray2, (7, 7), cv.BORDER_DEFAULT)                                                            # blur to reduce small edges/noise
    boundaries = cv.Canny(blur, 40, 60)                                                                                 # edge detection
    contours, hierarchies = cv.findContours(boundaries, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)                           # convert edges to contours
    filteredContours = []
    for contour in contours:
        L = cv.arcLength(contour, True)                                                                                 # filter each contour by arcLength
        if L >= ARC_LENGTH_THRESHOLD:
            newContour = cv.approxPolyDP(contour, int(L * TRACE_ROUGHNESS_FACTOR), True)                                # then simplify curve
            if cv.arcLength(newContour, True) > ARC_LENGTH_THRESHOLD:
                filteredContours.append(newContour)                                                                     # if still long enough: save contour
    contourField = np.zeros(frame.shape, dtype="uint8")                                                                 # new canvas for contours
    # cv.drawContours(contourField, contours, -1, (255, 0, 255), 2)                                                       # draw contours
    cv.drawContours(contourField, filteredContours, -1, (255, 0, 255), 2)
    traceField = np.zeros(frame.shape, dtype="uint8")                                                                   # new canvas for isolated shapes
    ARframe = frame.copy()                                                                                              # copy frame for final output
    located = 0                                                                                                         # found shapes counter
    # for contour in contours:
    for contour in filteredContours:
        # trace = cv.approxPolyDP(contour, TRACE_ROUGHNESS, True)
        trace = contour
        L = cv.arcLength(contour, True)                                                                                 # arc length of current contour
        if L >= ARC_LENGTH_THRESHOLD:                                                                                   # ignore small contours
            # trace = cv.approxPolyDP(contour, TRACE_ROUGHNESS, True)                                                     # approximate shape from contour
            # if 3 <= len(trace) <= 4:                                                                                    # ignore shapes without 3-4 sides
            if len(trace) > 1:
                sidesConf = fuzzifyNumSides(len(trace))
                cv.drawContours(traceField, [trace], 0, (255, 255, 255), 3)                                             # draw shape to shape canvas
                x, y, w, h, = cv.boundingRect(trace)                                                                    # get rectangle parameters
                A = float(w)/float(h)
                aspectConf = fuzzifyApsect(A)
                confidence = sidesConf * aspectConf
                if confidence >= 0.01:
                    located += 1
                    prevRect = (x, y, w, h)                                                                             # save good rectangle parameters
                    cv.putText(traceField, str(L), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)# put label on shape canvas
                    cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                             # draw bounding rectangle on output
                    cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)           # draw 'name tab' rectangle on output
                    cv.putText(ARframe, "goal " + str(confidence), (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 2)      # put label in 'name tab' on output
    if located == 0:
        stagnationIn += 1                                                                                               # if no shapes found, increment counter
        if stagnationIn < STAGNATION_LIMIT:
            x, y, w, h, = prevRect                                                                                      # if within limit, use last good rect
            f = max(0, 255 - int(stagnationIn * (255 / STAGNATION_LIMIT)))                                              # 'fading' font color for shape canvas
            cv.putText(traceField, "•", (x + w, y + h), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (f, f, f), 2)               # put label at last good location
            cv.rectangle(ARframe, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)                                     # redraw last good rectangles on output
            cv.rectangle(ARframe, (x - 1, y + h), (x + w + 1, y + h + 20), (0, 0, 255), thickness=-1)
            cv.putText(ARframe, "goal", (x + 2, y + h + 12), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 2)          # redraw last good label on output
    else:
        stagnationIn = 0                                                                                                # if >0 shapes found, reset counter

    # cv.imshow("masked", masked)                                                                                         # DEBUG: HSV masked output
    cv.imshow("threshmasked", threshmasked)                                                                             # DEBUG: HSV/threshold masked output
    cv.imshow("Canny", boundaries)                                                                                      # DEBUG: edge detection output
    cv.imshow("contours", contourField)                                                                                 # DEBUG: contour output
    cv.imshow("shape traces", traceField)                                                                               # DEBUG: traced shapes output
    cv.imshow("goal locator", ARframe)                                                                                  # final "augmented reality" output
    return located, prevRect, stagnationIn


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
        return gaussian(n, 4, 2.5, 'z')


# def dance():
#     crawler.do_action('forward', 2, SPEED)
#     crawler.do_action('look_left', 1, SPEED)
#     crawler.do_action('look_right', 1, SPEED)
#     for count in range(2):
#         crawler.do_action('look_left', 1, SPEED)
#         crawler.do_action('look_right', 1, SPEED)
#     for count2 in range(3):
#         crawler.do_action('sit', 1, SPEED)
#         crawler.do_action('stand', 1, SPEED)
#     crawler.do_action('push up', 1, SPEED)
#     crawler.do_action('backward', 2, SPEED)
#     crawler.do_action('dance', 1, SPEED)


if __name__ == "__main__":
    cam = cv.VideoCapture(0)                                                                                            # instantiate cam feed
    cam.set(3, FEED_RES_W)                                                                                              # set feed resolution
    cam.set(4, FEED_RES_H)
    previousRect = (0, 0, 0, 0)                                                                                         # storage to remember rectangle
    stagnation = 0                                                                                                      # counter to hold "lost" shapes

    loop = True                                                                                                         # boolean loop variable
    while loop:
        prevStagnation = stagnation
        located, previousRect, stagnation = showFeed(cam, previousRect, stagnation)
        if located > 0:
            # crawler.do_action("forward", 2, SPEED)
            time.sleep(0.01)
        else:
            # crawler.do_action("turn left", 1, SPEED)
            time.sleep(0.01)

        if cv.waitKey(1) & 0xFF == ord('q'):
            loop = False
