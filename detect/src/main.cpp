/* 
	Headless Bottle Detector used for quicker
	speeds on robot.
*/

#include <string>
#include <iostream>
#include <vector>
#include <iomanip>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "gpio.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

/* Constants */

const string labels = "mob/labels.txt";
const string model = "mob/model.xml";
const string weights = "mob/model.bin";

const float conf_threshold = 0.4;
float confidence;

int main() {
	
	double time;
	vector<double> layerTimes;

	int deviceID = 0;
	Mat frame;
	VideoCapture cap;
	cap.open(deviceID);

	/* Init GPIO - function speaks for itself */

	init_gpio();

	/* ML network variables */

	cout << "Loading model." << endl;

	dnn::Net net = readNetFromModelOptimizer(model, weights);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_MYRIAD);
	
	/* Detection loop */

	cout << "Detection loop is initializing." << endl;
	
	while(1) {
		cap.read(frame);

		Mat blob = dnn::blobFromImage(frame, 1, Size(640, 640));

		net.setInput(blob);
		Mat preds = net.forward();
		Mat matrix(preds.size[2], preds.size[3], CV_32F, preds.ptr<float>());

		for (int i = 0; i < matrix.rows; i++) {
			confidence = matrix.at<float>(i, 2);
			if (confidence < conf_threshold) {
				continue;
			}
			gpio_on();
			time = net.getPerfProfile(layerTimes) / getTickFrequency() / 1000.0;
			cout << "confidence:\t" << confidence << "\tlatency:\t" << time << endl;
		}
	}
	gpio_off();
	return 0;
}
