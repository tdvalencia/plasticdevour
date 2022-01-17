from openvino.inference_engine import IECore
import cv2
import numpy as np

# Collect data
image = cv2.imread('IMG_0565.JPG')
resized = cv2.resize(image, (640, 640))
input_data = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0).astype(np.float16)

ie = IECore()

def main():
	net = ie.read_network(model=model_path)

	inputs = net.input_info
	input_name = next(iter(net.input_info))
	net.input_info[input_name].precision = 'FP16'

	outputs = net.outputs
	output_name = next(iter(net.outputs))

	exec_net = ie.load_network(network=net, device_name='MYRIAD')
	print('[INFO] Loaded Network')

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('cannot open camera')
		exit()

	while True:
		ret, frame = cap.read
		print('[INFO] inference')
		result = exec_net.infer({input_name: input_data})
		print('[INFO] complete')

		outputs = result[output_name]

def stop():
	cap.release()
	cv.DestroyAllWindows()
