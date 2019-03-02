from centroidtrack.centroidtracker import CentroidTracker
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

ct = CentroidTracker()
(H, W) = (None, None)

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("starting video stream...")
vs = cv2.VideoCapture("Dance.mp4")

while True:
	ret, frame = vs.read()
	small_frame = cv2.resize(frame, (0, 0), fx=1.80, fy=1.80)
	rgb_small_frame = small_frame[:, :, ::-1]

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	for i in range(0, detections.shape[2]):

		if detections[0, 0, i, 2] > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))
			(startX, startY, endX, endY) = box.astype("int")
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

	objects = ct.update(rects)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()