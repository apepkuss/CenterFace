import cv2

import os
from centerface import CenterFace


def test_image_tensorrt():
    frame = cv2.imread("prj-python/000388.jpg")
    # * set height and width
    h, w = 480, 640  # must be 480* 640
    landmarks = False
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, h, w, threshold=0.35)
    print("count = ", len(dets))

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(
            frame,
            (int(boxes[0]), int(boxes[1])),
            (int(boxes[2]), int(boxes[3])),
            (2, 255, 0),
            1,
        )
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(
                    frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1
                )
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite("res_000388.jpg", frame)
    # cv2.imshow("out", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_image_tensorrt()
