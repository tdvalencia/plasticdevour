/* 
	Headless Bottle Detector used for quicker
	speeds on robot.
*/

#include <ctime>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "DEV_Config.h"
#include <time.h>
#include "MotorDriver.h"

#include <signal.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::this_thread;
using namespace std::chrono;
using namespace cv;
using namespace cv::dnn;

/* Constants */

const string labels = "mob/labels.txt";
const string model = "mob/model.xml";
const string weights = "mob/model.bin";

const int waitTime = 2;
const float conf_threshold = 0.5;
float confidence;

void Handler(int signo) {
	//System Exit
    printf("\r\nHandler: Motor Stop\r\n");
    Motor_Stop(MOTORA);
    DEV_ModuleExit();

    exit(0);
}

int main() {
	
	double latency;
	vector<double> layerTimes;

	int deviceID = 0;
	Mat frame;
	VideoCapture cap;
	cap.open(deviceID);

	/* Init GPIO - function speaks for itself */

	//if (DEV_ModuleInit()) {
	// 	return 1;
	//}
	//Motor_Init();

	/* ML network variables */

	cout << "Loading model." << endl;

	dnn::Net net = readNetFromModelOptimizer(model, weights);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_MYRIAD);
	
	/* Detection loop */

	cout << "Detection loop is initializing." << endl;

	freopen("output.csv", "w", stdout);
	cout << "confidence,latency" << endl;
	// signal(SIGINT, Handler);
	while(1) {
		cap.read(frame);

		Mat blob = dnn::blobFromImage(frame, 1, Size(640, 640));

		net.setInput(blob);
		Mat preds = net.forward();
		Mat matrix(preds.size[2], preds.size[3], CV_32F, preds.ptr<float>());

		for (int i = 0; i < matrix.rows; i++) {
			confidence = matrix.at<float>(i, 2);
			if (confidence < conf_threshold) {
				//sleep_until(system_clock::now() + seconds(waitTime));
				//Motor_Stop(MOTORA);
				continue;
			}
			//Motor_Run(MOTORA, BACKWARD, 100);
			latency = net.getPerfProfile(layerTimes) / getTickFrequency() / 1000.0;
			cout << confidence << "," << latency << time(0) << endl;
		}
		//int key = waitKey(10);
		//if((char)key == 'c') { break; }
	}
	//Motor_Stop(MOTORA);
	//DEV_ModuleExit();
	return 0;
}
