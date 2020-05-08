import cv2

import os
from centerface import CenterFace


def test_image_tensorrt():
    frame = cv2.imread("/home/nano/workspace/CenterFace/prj-python/000388.jpg")
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


def test_cam_tensorrt():
    uri = "rtsp://192.168.1.89/user=admin&password=&channel=1&stream=0.sdp?"
    width = 1920
    height = 1080
    latency = 20
    cam = open_gst_camera(uri, width, height, latency)
    ok, frame = cam.read()

    while ok:
        landmarks = False
        centerface = CenterFace(landmarks=landmarks)
        if landmarks:
            dets, lms = centerface(frame, 480, 640, threshold=0.35)
        else:
            dets = centerface(frame, 480, 640, threshold=0.35)
        print("count = ", len(dets))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        ok, frame = cam.read()


def open_gst_camera(
    uri: str, width: int, height: int, latency: int
) -> cv2.VideoCapture:
    assert uri is not None, "'uri' should not be None."
    assert isinstance(uri, str), "'uri' should be of str type."
    assert width is not None, "'width' should not be None."
    assert isinstance(width, int), "'width' should be of int type."
    assert height is not None, "'height' should not be None."
    assert isinstance(height, int), "'height' should be of int type."
    assert latency is not None, "'latency' should not be None."
    assert isinstance(latency, int), "'latency' should be of int type."

    gst_str = (
        "rtspsrc location={} latency={} ! "
        "rtph264depay ! h264parse ! omxh264dec ! "
        "nvvidconv ! "
        "video/x-raw, width=(int){}, height=(int){}, "
        "format=(string)BGRx ! "
        "videoconvert ! appsink"
    ).format(uri, latency, width, height)
    print(f"[INFO] gst_str: {gst_str}")

    # create VideoCapture object
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


if __name__ == "__main__":
    # test_image_tensorrt()
    test_cam_tensorrt()
