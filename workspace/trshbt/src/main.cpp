#include "main.h"

#include <map>
#include <string>
#include <string.h>

#define model_path "/home/pi/dev/trshbt/workspace/openvino/mob/ncs2/model.xml"

using namespace InferenceEngine;

int main(int argc, char **argv) {

	std::string filename;

	if (argc < 3) {
		std::cout << "Too many args: use -i 'image.jpg'." << std::endl;
		return 1;
	}

	if (strcmp(argv[1], "-i") == 0) {
		filename = argv[2];
	}

	Core core;
	CNNNetwork network = core.ReadNetwork(model_path);

	/** Config input params **/
	InputsDataMap inputInfo = network.getInputsInfo();
	for (auto &item : inputInfo) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP16);
	}

	/** Config output params **/
	OutputsDataMap outputInfo = network.getOutputsInfo();
	for (auto &item : outputInfo) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP16);
	}

	/** Load the network **/
	ExecutableNetwork executable_network = core.LoadNetwork(network, "MYRIAD");
	auto infer_request = executable_network.CreateInferRequest();

	/** Preparing the input **/
	for (auto &item : inputInfo) {
		auto input_name = item.first;
		auto input = infer_request.GetBlob(input_name);
	}

	/** Inference (the real detection) **/
	infer_request.Infer();

	/** Process inference **/
	for (auto &item : outputInfo) {
		auto output_name = item.first;
		auto output = infer_request.GetBlob(output_name);
		{
			auto const memLocker = output->cbuffer();
			const float *output_buffer = memLocker.as<const float *>();

		}
	}

	std::cout << "it's trashboat time!!" << std::endl;

	return 0;
}

