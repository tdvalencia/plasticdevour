from openvino.inference_engine import IECore
import cv2
import numpy as np

# Collect data
image = cv2.imread('img.jpg')
resized = cv2.resize(image, (640, 640))
input_data = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0).astype(np.float16)

# Set up model
ie = IECore()
net = ie.read_network(model='model.xml')

inputs = net.input_info
input_name = next(iter(net.input_info))
net.input_info[input_name].precision = 'FP16'

print("Inputs:")
for name, info in net.input_info.items():
    print("\tname: {}".format(name))
    print("\tshape: {}".format(info.tensor_desc.dims))
    print("\tlayout: {}".format(info.layout))
    print("\tprecision: {}\n".format(info.precision))

outputs = net.outputs
output_name = next(iter(net.outputs))

print("Outputs:")
for name, info in net.outputs.items():
    print("\tname: {}".format(name))
    print("\tshape: {}".format(info.shape))
    print("\tlayout: {}".format(info.layout))
    print("\tprecision: {}\n".format(info.precision))

exec_net = ie.load_network(network=net, device_name='MYRIAD')

# Inference
result = exec_net.infer({input_name: input_data})

outputs = result[output_name]

print("Outputs:")
for name, info in net.outputs.items():
	print("\tname: {}".format(name))
	print("\tshape: {}".format(info.shape))
	print("\tlayout: {}".format(info.layout))
	print("\tprecision: {}\n".format(info.precision))
