#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>


int PIXEL_PER_CM = 18;

using namespace std;
using namespace cv;

Mat soble_edge_detection(const Mat& frame)
{
	Mat gray, blurred, sobel_x, sobel_y, sobel_combined, edge, threshold_edge;

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blurred, Size(), 0);
	Sobel(blurred, sobel_x, CV_64F, 1, 0);
	Sobel(blurred, sobel_y, CV_64F, 0, 1);
	magnitude(sobel_x, sobel_y, sobel_combined);
	sobel_combined.convertTo(edge, CV_8U);
	
	Mat thredhold_edge = edge > 80;
	
	return thredhold_edge;
}

bool is_cal_box(Mat frame)
{
	bool is_cal_box = false;
	int low_H = 7, high_H = 64;
	int low_S = 45, high_S = 212;
	int low_V = 0, high_V = 255;

	Mat frame_HSV, frame_threshold;
	cvtColor(frame, frame_HSV, COLOR_BGR2GRAY);
	inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V), frame_threshold);

	vector<vector<Point>> contours;
	findContours(frame_threshold, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	float max_area = 0.0;
	size_t max_area_index = -1;

	for (size_t i = 0; i < contours.size(); ++i) {
		vector<Point> hull;
		convexHull(contours[i], hull);

		double area;
		area = contourArea(hull);
		if (area < 4.0)
			continue;
		if (max_area < area) {
			max_area = area;
			max_area_index = i;
		}
	}
	
	if (max_area_index != -1) {
		
		vector<Point> contour;
		contour = contours[max_area_index];
				
		RotatedRect min_rect;
		min_rect = minAreaRect(contour);

		int x, y, w, h;

		x = static_cast<int>(min_rect.center.x);
		y = static_cast<int>(min_rect.center.y);
		w = static_cast<int>(min_rect.size.width);
		h = static_cast<int>(min_rect.size.height);
				
		int dist = 50;		
		line(frame_HSV, Point(x - dist, y - dist), Point(x + dist, y + dist), (0, 0, 255), 5);
		line(frame_HSV, Point(x + dist, y - dist), Point(x - dist, y + dist), (0, 0, 255), 5);

		int roi_start_x, roi_end_x, roi_start_y, roi_end_y;
		roi_start_x = max(0, x - 10);
		roi_start_y = max(0, y - 10);
		roi_end_x = min(frame_HSV.cols, x + 10);
		roi_end_y = min(frame_HSV.rows, y + 10);

		Rect roi(roi_start_x, roi_start_y, roi_end_x - roi_start_x, roi_end_y - roi_start_y);
		Mat frame_roi;
		frame_roi = frame_HSV(roi);
		
		w = max(w, h);
		h = min(w, h);
		PIXEL_PER_CM = static_cast<double>(w) / 20.0;
		
		Scalar roi_mean;
		roi_mean = mean(frame_roi);

		if (roi_mean[1] > 20 && roi_mean[1] < 50) {
			is_cal_box = true;
		}
	}

	return is_cal_box;
}

Mat measure_and_draw_boxes(Mat frame, Mat edged_frame)
{
	vector<vector<Point>> contours;
	findContours(edged_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	float max_area = 0.0;
	int max_area_index = -1;

	for (size_t i = 0; i < contours.size(); ++i) {
		vector<Point> hull;
		convexHull(contours[i], hull);

		double area;
		area = contourArea(hull);
		if (area < 4)
			continue;
		if (max_area < area) {
			max_area = area;
			max_area_index = i;
		}
	}


	if (max_area_index != -1 && is_cal_box(frame)) {
		vector<Point> contour;
		contour = contours[max_area_index];

		RotatedRect min_rect;
		min_rect = minAreaRect(contour);

		int x, y, w, h;

		x = static_cast<int>(min_rect.center.x);
		y = static_cast<int>(min_rect.center.y);
		w = static_cast<int>(min_rect.size.width);
		h = static_cast<int>(min_rect.size.height);

		w = max(w, h);
		h = min(w, h);

		uint16_t box_B, box_B_w, box_M, box_M_w, box_S, box_S_w;

		box_B = 26;
		box_B_w = 400;
		box_M = 20;
		box_M_w = 300;
		box_S = 16;
		box_S_w = 200;

		if (w > box_B_w)
			PIXEL_PER_CM = w / box_B;
		else if (w > box_M_w)
			PIXEL_PER_CM = w / box_M;
		else if (w > box_S_w)
			PIXEL_PER_CM = w / box_S;
	}

	for (size_t i = 0; i < contours.size(); ++i) {
		double area;
		area = contourArea(contours[i]);
		if (area > 100)
			continue;

		vector<Point> hull;
		convexHull(contours[i], hull);
		
		Rect bounding_rect = boundingRect(hull);
		int x, y, w, h;

		x = bounding_rect.x;
		y = bounding_rect.y;
		w = bounding_rect.width;
		h = bounding_rect.height;

		double width_cm, height_cm, box_area_cm2;
		width_cm = static_cast<double>(w) / PIXEL_PER_CM;
		height_cm = static_cast<double>(h) / PIXEL_PER_CM;
		box_area_cm2 = sqrt(pow(width_cm, 2) + pow(height_cm, 2));

		string box_category;
		if (box_area_cm2 > 27)
			box_category = "L";
		else if (box_area_cm2 >= 22 && box_area_cm2 <= 27)
			box_category = "M";
		else
			box_category = "S";

		if (width_cm <= 8 && height_cm <= 8)
			continue;

		cv::rectangle(frame, bounding_rect, Scalar(0, 255, 0), 2);

		string text = format("width: %.2fcm X height: %.2fcm (size: %s)", width_cm, height_cm, box_category);
		int baseline = 0;

		Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
		Point text_start_point(x, y - text_size.height - baseline);

		cv::rectangle(frame, text_start_point + Point(0, baseline), text_start_point + Point(text_size.w, -text_size.height), (0, 0, 0), FILLED);
		cv::putText(frame, text, text_start_point, FONT_HERSHEY_SIMPLEX, 0.5, 1, (0, 255, 0), 1);
	}
	return frame;
}



int main(void)
{
	cout << "Hello OpenCV " << CV_VERSION << endl;

	Mat img;
	img = imread("lena.jpg");

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("image");
	imshow("image", img);
	
	waitKey();
	return 0;
}