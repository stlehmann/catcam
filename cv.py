import base64
import cv2
import imutils
import numpy as np

from typing import List, Union, Tuple, NamedTuple, Optional


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    def get_area(self) -> int:
        """Return the area of the rectangle."""
        return self.w * self.h


RectUnion = Union[Tuple[int, int, int, int], Rect]

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
MIN_CONTOUR_AREA = 25


def process_image(input_frame: np.ndarray, dilate_iterations: int = 1) -> Tuple[np.ndarray, List[Rect]]:
    # convert to gray
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    # apply background substraction
    fgmask = fgbg.apply(gray)
    thresh_frame = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    if dilate_iterations:
        thresh_frame = cv2.erode(thresh_frame, None, iterations=dilate_iterations)
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=dilate_iterations)

    # find contours
    cnts = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    rects = combine_rects(rects, gap=10)
    rects = sorted(rects, key=lambda x: x.get_area(), reverse=True)

    return thresh_frame, rects


def rects_overlap(r1: Rect, r2: Rect, g: int = 0) -> bool:
    """Check if two rectangles overlap / touch each other.
    :param r1: first rectangle
    :param r2: second rectangle
    :param g: maximum gap between rectangles
    :return: True if rectangles overlap or border each other
    """
    x00, y00, x01, y01 = r1.x, r1.y, r1.x + r1.w, r1.y + r1.h
    x10, y10, x11, y11 = r2.x, r2.y, r2.x + r2.w, r2.y + r2.h

    return x10 - x01 <= g and x00 - x11 <= g and y10 - y01 <= g and y00 - y11 <= g


def combine_rects(rects_: List[RectUnion], gap: int = 0) -> List[Rect]:
    """Combine overlapping and adjacent rectangles to bigger rectangles.
    :param rects_: List of rectangles
    :param gap: Maximum gap between rectangles for combining
    :return: List of combined rectangles
    """
    rects: List[Rect] = [Rect(x, y, w, h) for (x, y, w, h) in rects_]

    # find overlapping rects
    overlap = True
    while overlap:
        overlap = False
        for r0 in rects:
            for r1 in rects:
                if r0 == r1:  # skip same rectangles
                    continue

                if rects_overlap(r0, r1, g=gap):
                    rects.remove(r0)
                    rects.remove(r1)
                    rects.append(create_combined_rect(r0, r1))
                    overlap = True
                    break
            if overlap:
                break
    return rects


def create_combined_rect(r0: Rect, r1: Rect) -> Rect:
    """Combine two rectangles to one containing both."""
    x0, y0, w0, h0 = r0
    x1, y1, w1, h1 = r1
    new_x0, new_y0 = min(x0, x1), min(y0, y1)
    new_x1, new_y1 = max(x0 + w0, x1 + w1), max(y0 + h0, y1 + h1)
    return Rect(new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0)


def frame_to_base64(frame_: np.ndarray) -> str:
    """Convert an image frame (numpy array) to base64-encoded image data."""
    _, frame = cv2.imencode(".jpg", frame_)
    data = base64.b64encode(bytearray(frame))
    return "data:image/jpeg;base64," + data.decode()