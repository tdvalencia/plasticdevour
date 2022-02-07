import cv2
import time
import numpy as np

# Params
model_path = "../mob/model.bin"
arch_path = "../mob/model.xml"
thres = 0.5

# Load model
net = cv2.dnn.readNet(arch_path, model_path)

# Specify NCS2
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD) # DNN_TARGET_CPU (only if you don't have NCS2)

vid_cap = cv2.VideoCapture(0)
if not vid_cap.isOpened():
	raise IOError("Webcam not opened")

while True:
	# Capture frame
	ret, frame = vid_cap.read()

	# Prepare input blob and preform inference
	blob = cv2.dnn.blobFromImage(frame, size=(640, 640), ddepth=cv2.CV_8U)
	net.setInput(blob)
	start = time.time()
	out = net.forward()
	end = time.time()

	for detect in out.reshape(-1, 7):
		conf = float(detect[2])

		if conf > thres:
			lat = str((end-start)*1000)
			print(f"confidence {conf}\tlatency {lat} ms")

	# ESC to quit
	if cv2.waitKey(1) == 27:
		break

# Relase vid_cap and close windows
vid_cap.release()
cv2.destroyAllWindows()
