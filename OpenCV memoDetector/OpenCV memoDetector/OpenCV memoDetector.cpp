#include <opencv2/opencv.hpp>
#include <iostream>
#include<thread>

#include<io.h>

using namespace cv;
using namespace std;

cv::Mat getHist(cv::Mat image);
cv::Mat Rotation(cv::Mat src);

Mat PerspectiveTransform(Mat src, vector<Point>& square);
Mat DrawSquare(Mat src, vector<Point>& square);
Mat getMarker(Mat src, vector<Point>& square);
double cosAngle(Point pt1, Point pt0, Point pt2);
vector<Point> Square(Mat src);
void findBiggestSquare(Mat src, vector<Point>& square, double& maxArea);

cv::Mat findPostIt(cv::Mat imgOrigin, cv::Mat img);

std::string fileNames[] = {
  "imgs/IMG_1521_3.jpg",					// 0
  "imgs/IMG_1523_2.jpg",
  "imgs/demo2_output/IMG_1519_3.jpg",
  "imgs/demo2_output/IMG_1540_5.jpg",
  "imgs/demo2_output/IMG_1523_4.jpg",
  "imgs/demo2_output/IMG_1544_3.jpg",		// 5
  "imgs/demo2_output/IMG_1542_4.jpg",
  "imgs/demo2_output/IMG_1546_8.jpg",
  "imgs/demo2_output/IMG_1546_12.jpg",
  "imgs/demo2_output/IMG_1545_5.jpg",
  "imgs/demo2_output/IMG_1544_1.jpg",		// 10
  "imgs/demo2_output/IMG_1544_4.jpg",
  "imgs/KakaoTalk_20230516_172222664.jpg",
  "imgs/KakaoTalk_20230516_172222874.jpg",
  "imgs/KakaoTalk_20230516_172222797.jpg",
  "imgs/KakaoTalk_20230516_172222726.jpg"	// 15
};

vector<string> fileNames2;

typedef struct threadParameter {
	vector<cv::Mat>* resized;
	vector<cv::Mat>* dst;
	int n;
}threadParameter;

void getResult(threadParameter tp);

bool debug = false;

int main()
{
	//int fileN = sizeof(fileNames) / sizeof(string);

	string path = ".\\imgs\\*.jpg";

	struct _finddata_t fd;

	intptr_t handle;

	if ((handle = _findfirst(path.c_str(), &fd)) == -1L)
		cout << "No file in directory!" << endl;

	int i = 0;
	do {
		fileNames2.push_back(fd.name);
	} while (_findnext(handle, &fd) == 0);

	_findclose(handle);

	int fileN = fileNames2.size();

	vector<cv::Mat> src(fileN);
	vector<cv::Mat> results(fileN);
	vector<thread> threads(fileN);

	threadParameter tp;
	tp.dst = &results;
	tp.resized = &src;

	debug = true;

	if (debug) {
		int m = 3;
		tp.n = m;
		thread th = thread(&getResult, tp);
		th.join();
		imshow("result", results[m]);
	}
	else {
		// 쓰레드 병렬 처리
		for (int i = 0; i < fileN; i++) {
			tp.n = i;
			threads[i] = thread(&getResult, tp);
		}

		// 모든 쓰레드가 종료될 때까지 대기
		for (int i = 0; i < fileN; i++) {
			cout << i << " / " << fileN << " completed" << endl;
			threads[i].join();
		}

		//show result image
		for (int i = 0; i < fileN; i++) {
			// show resized image
			//imshow(to_string(i) + "th img", src[i]);
			//show result image
			//imshow(to_string(i) + "th result", results[i]);

			imwrite("./src/" + to_string(i) + "th src.jpg", src[i]);
			imwrite("./results/" + to_string(i) + "th result.jpg", results[i]);
		}
	}
	cv::waitKey(0);

	return 0;
}

void getResult(threadParameter tp) {
	vector<cv::Mat>& src = *tp.resized;
	vector<cv::Mat>& results = *tp.dst;
	int n = tp.n;

	// read imageFile
	cv::Mat input = imread(".\\imgs\\" + fileNames2[n]);
	//cv::Mat input = imread(fileNames[n]);

	// resizing
	double ratio = 800.0 / input.cols;
	int width = input.cols * ratio;
	int height = input.rows * ratio;
	cv::Size size(width, height);

	Mat resized;
	cv::resize(input, resized, size);

	// resized 이미지 저장
	src[n] = resized;

	// process
	cv::Mat result = findPostIt(input, resized);

	// 결과 이미지 저장
	results[n] = result;

	if (debug) {
		imshow("result", result);
		waitKey(0);
	}

	return;
}

cv::Mat findPostIt(cv::Mat imgOrigin, cv::Mat img) {

	cv::Mat imgHSV;
	cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

	if (debug) imshow("img1HSV", imgHSV);

	vector<Point> square = Square(imgHSV);
	if (square.size() != 4) {
		/*
		// 크기를 4분의 1로 줄여나가며 사각영역을 찾음
		int n = 1;
		while (square.size() != 4) {
			resize(imgHSV, imgHSV, Size(imgHSV.cols / 2, imgHSV.rows / 2));
			square = Square(imgHSV);
			n *= 2;
		}
		if (square.size() == 4) {
			square[0] *= n;
			square[1] *= n;
			square[2] *= n;
			square[3] *= n;
		}
		// 여전히 사각 영역을 찾지 못했을 경우
		else */ {
			square.push_back(Point(50, 50));
			square.push_back(Point(img.cols - 50, 50));
			square.push_back(Point(img.cols - 50, img.rows - 50));
			square.push_back(Point(50, img.rows - 50));
		}
	}
	if (debug) {
		Mat squareImg = DrawSquare(img, square);
		imshow("square", squareImg);
	}

	// 마커 이미지 생성
	Mat marker = getMarker(img, square);
	if (debug) imshow("marker", marker);

	// 워터쉐드 알고리즘
	cv::Mat waterShed;
	marker.convertTo(waterShed, CV_32S);

	Mat gaussian1, gaussian2;
	GaussianBlur(img, gaussian1, Size(5, 5), 1.6);
	GaussianBlur(img, gaussian2, Size(5, 5), 1.0);
	Mat dog = gaussian1 - gaussian2;
	threshold(dog, dog, 0, 255, THRESH_BINARY);

	double r = 10.0 / img.cols;
	Size resizeBlur(img.cols*r, img.rows*r);
	Mat Blur;
	resize(img, Blur, resizeBlur);
	resize(Blur, Blur, Size(img.cols, img.rows));

	cv::watershed(Blur+dog, waterShed);

	// 생성된 포스트잇 영역 threshold 처리
	waterShed.convertTo(waterShed, CV_8UC1);
	rectangle(waterShed, Rect(0, 0, waterShed.cols, waterShed.rows), Scalar(255));
	Laplacian(waterShed, waterShed, waterShed.depth());

	if (debug) imshow("watershed", waterShed);

	Mat thres;
	threshold(waterShed, thres, 0, 255, THRESH_BINARY_INV);
	//erode(thres, thres, Mat());

	if (debug) imshow("thres", thres);

	//======================= hough 변환 이용===================================

	std::vector<cv::Vec2f> lines;
	std::vector<cv::Vec2f> lines2;

	cv::HoughLines(255 - thres, lines, 1, CV_PI / 180, 50);
	cv::HoughLines(255 - thres, lines2, 1, CV_PI / 180, 100);

	uint linesSize = lines2.size();

	lines.insert(lines.begin(), lines2.begin(), lines2.end());

	uint linesSizeTotal = lines.size();

	cv::Mat empty = Mat(thres.size(), CV_8UC1, Scalar(0));
	cv::Mat hough = empty.clone();

	Point middle(empty.cols / 2, empty.rows / 2);

	int w = 20;
	for (uint i = 0; i < linesSizeTotal; i++) {

		if (linesSize <= i) w = 10;

		double rho = lines[i][0];
		double theta = lines[i][1];

		// sin(theta) => 1이 되어 오류 발생
		if (theta - 1.5707 < 0.0001) theta += 0.001;

		cv::Point p1, p2;

		//====== p1 ==========
		// y = 0
		p1.y = 0;
		p1.x = cvRound((rho - p1.y * sin(theta)) / cos(theta));

		// x = 0
		if (p1.x < 0) {
			p1.x = 0;
			p1.y = (rho - p1.x * cos(theta)) / sin(theta);
		}
		// x = img.cols-1
		else if (hough.cols < p1.x) {
			p1.x = hough.cols - 1;
			p1.y = (rho - p1.x * cos(theta)) / sin(theta);
		}

		//====== p2 ==========
		// y = image.rows-1
		p2.y = hough.rows - 1;
		p2.x = cvRound((rho - p2.y * sin(theta)) / cos(theta));

		// x = 0 
		if (p2.x < 0) {
			p2.x = 0;
			p2.y = (rho - p2.x * cos(theta)) / sin(theta);
		}
		// x = img.cols - 1
		else if (hough.cols < p2.x) {
			p2.x = hough.cols - 1;
			p2.y = (rho - p2.x * cos(theta)) / sin(theta);
		}

		//============= p1 p2 좌표 구하기 완료=====================

		cv::Mat emptyClone = empty.clone();
		line(emptyClone, p1, p2, Scalar(w), 3);
		hough += emptyClone;
	}

	if (debug) imshow("hough", hough);
	double maxArea = 0.0;

	for (int i = 0; i < 225; i++) {
		cv::Mat thres = hough.clone();
		threshold(thres, thres, i, 255, cv::THRESH_BINARY);

		if (debug) {
			imshow("thresHough", thres);
			//waitKey(0);
		}

		// close 연산
		dilate(thres, thres, cv::Mat(), Point(-1, -1), 5);
		erode(thres, thres, cv::Mat(), Point(-1, -1), 5);

		rectangle(thres, Rect(0, 0, thres.cols, thres.rows), Scalar(100), 1);

		findBiggestSquare(thres, square, maxArea);
	}


	cv::Mat resultSquare = DrawSquare(img, square);
	if (debug) imshow("resultSquare", resultSquare);

	// 얻어낸 좌표를 토대로 perspective 변환 
	double ratio = 800.0 / imgOrigin.cols;
	square[0] /= ratio;
	square[1] /= ratio;
	square[2] /= ratio;
	square[3] /= ratio;
	Mat dst = PerspectiveTransform(imgOrigin, square);

	if (debug) imshow("dst", dst);

	// 마크를 기준으로 이미지 회전
	cv::Mat rotated = Rotation(dst);

	return rotated;
}

//=====================================================================================
cv::Mat Rotation(cv::Mat src) {
	//==============mark 처리===============================================

	const int markSize = 400;
	const int markWH = 70; // 800*800 이미지에서의 적당한 mark 검출 영역

	const int markInWH = 230;
	const int markInStart = 85;

	//============== 원본영상 처리==========================================

	// gray 영상 얻기
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	//resizing
	// 원본 이미지의 mark 부분 크기가, markSize가 되도록 원본 이미지 사이즈 조정
	double ratio = (markSize / markWH);
	int width = gray.cols * ratio;
	int height = gray.rows * ratio;
	cv::Size size(width, height);

	cv::resize(gray, gray, size);

	// 선명도 높임
	float sharpen[] = {
		-1,-1,-1,
		-1, 9,-1,
		-1,-1,-1
	};
	cv::Mat kernel(Size(3, 3), CV_32FC1, sharpen);

	filter2D(gray, gray, gray.depth(), kernel * 2);

	//===================================================================================

	// Rotation Test
	int rotate;
	bool flag;

	const int startMarkX = 400;
	const int startMarkY = 400;

	for (rotate = 0; rotate < 4; rotate++) {
		flag = true;

		if (debug) imshow("gray", gray);

		// 마크 예상 영역 분리
		cv::Mat ROI = gray(Rect(10, 10, 800, 800)).clone();

		// 이진 이미지로 변환
		cv::threshold(ROI, ROI, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		rectangle(ROI, Rect(0, 0, 800, 800), Scalar(255), 3);

		// binary_inv
		bitwise_not(ROI, ROI);

		//================ 세로방향 보완 ======================================
		cv::Mat empty = Mat(ROI.size(), CV_8UC1, Scalar(0));
		line(empty, Point(3, 3), Point(3, 800 - 4), Scalar(255), 3);
		line(empty, Point(800 - 4, 3), Point(800 - 4, 800 - 4), Scalar(255), 3);
		bitwise_and(ROI, empty, empty);

		// 연결이 끊어진 부분 보완
		bool isConnectedL = false, isConnectedR = false;
		int startL = -1, startR = -1;
		for (int i = 0; i < 800; i++) {
			uchar* emptyPtr = empty.ptr<uchar>(i);

			// 왼쪽 부분
			if (emptyPtr[3] == 255) {
				if (isConnectedL != true) {
					if (startL != -1 && i - startL < 100)
						line(empty, Point(3, startL), Point(3, i), Scalar(255), 3);
					isConnectedL = true;
				}
				startL = i;
			}
			else isConnectedL = false;

			// 오른쪽 부분
			if (emptyPtr[800 - 4] == 255) {
				if (isConnectedR != true) {
					if (startR != -1 && i - startR < 100)
						line(empty, Point(800 - 4, startR), Point(800 - 4, i), Scalar(255), 3);
					isConnectedR = true;
				}
				startR = i;
			}
			else isConnectedR = false;
		}

		ROI += empty;

		//================ 가로방향 보완 ======================================
		empty = Mat(ROI.size(), CV_8UC1, Scalar(0));
		line(empty, Point(3, 3), Point(800 - 4, 3), Scalar(255), 3);
		line(empty, Point(3, 800 - 4), Point(800 - 4, 800 - 4), Scalar(255), 3);
		bitwise_and(ROI, empty, empty);

		// 연결이 끊어진 부분 보완
		bool isConnectedU = false, isConnectedD = false;
		int startU = -1, startD = -1;

		uchar* emptyPtrU = empty.ptr<uchar>(3);
		uchar* emptyPtrD = empty.ptr<uchar>(800 - 4);
		for (int i = 0; i < 800; i++) {

			// 윗 부분
			if (emptyPtrU[i] == 255) {
				if (isConnectedU != true) {
					if (startU != -1 && i - startU < 100)
						line(empty, Point(startU, 3), Point(i, 3), Scalar(255), 3);
					isConnectedU = true;
				}
				startU = i;
			}
			else isConnectedU = false;

			// 아래 부분
			if (emptyPtrD[i] == 255) {
				if (isConnectedD != true) {
					if (startD != -1 && i - startD < 100)
						line(empty, Point(startD, 800 - 4), Point(i, 800 - 4), Scalar(255), 3);
					isConnectedD = true;
				}
				startD = i;
			}
			else isConnectedD = false;
		}

		ROI += empty;
		//=====================================================================


		if (debug)imshow("ROI", ~ROI);

		vector<vector<Point>> contours;
		vector<cv::Vec4i> hierarchy;
		cv::findContours(
			~ROI,
			contours,
			hierarchy,
			RETR_LIST,
			CHAIN_APPROX_TC89_KCOS
		);

		double min = ROI.cols * ROI.rows * 0.01;
		double max = ROI.cols * ROI.rows * 0.9;

		for (int i = 0; i < contours.size() && flag; i++) {

			double perimeter = arcLength(contours[i], true); // 윤곽선 전체 길이 계산, 닫힌곡선이므로 true
			vector<Point> result;
			approxPolyDP(contours[i], result, perimeter * 0.03, true); // 다각형 근사, 윤곽선 배열 반환

			double area = contourArea(result); // 면적판단
			bool convex = isContourConvex(result); // 다각형 형태인지 볼록성 판단

			if (debug && result.size() >= 4 && min < area && true) {
				cout << "max: " << max << endl;
				cout << "min: " << min << endl;
				cout << "area: " << area << endl;
				cv::Mat temp;
				cvtColor(ROI, temp, COLOR_GRAY2BGR);
				temp = DrawSquare(temp, result);
				imshow("temp", temp);
				waitKey(0);
			}

			if (result.size() == 4 && min < area && area < max && convex) {
				double cos = 0;
				for (int k = 0; k < 4; k++)
				{
					Point p1 = result[(k + 0) % 4];
					Point p0 = result[(k + 1) % 4];
					Point p2 = result[(k + 2) % 4];
					double t = cosAngle(p1, p0, p2);
					// t가 제일 큰, 즉 angle이 제일 작은 값을 찾아서 cos에 저장
					cos = cos > t ? cos : t;
				}

				// cos < 0.25
				// arccos(cos) > 75.52 degree
				// then isSquare = true
				if (cos < 0.25) {

					if (debug) {
						cv::Mat rect;
						cvtColor(ROI, rect, COLOR_GRAY2BGR);
						rect = DrawSquare(rect, result);
						imshow("ROIrect", rect);
					}

					cv::Mat ROI2 = PerspectiveTransform(~ROI, result);
					cv::threshold(ROI2, ROI2, 0, 255, cv::THRESH_OTSU);
					//rectangle(ROI, Rect(0, 0, 800, 800), Scalar(0), 3);

					if (debug) {
						imshow("ROI2", ROI2);
						//waitKey(0);
					}
					if (cv::sum(ROI2).val[0] / 255 / (ROI2.cols * ROI2.rows) < 0.3) {
						vector<vector<Point>> contours2;
						cv::findContours(
							ROI2,
							contours2,
							hierarchy,
							RETR_LIST,
							CHAIN_APPROX_TC89_KCOS
						);
						for (int j = 0; j < contours2.size() && flag; j++) {
							double perimeter = arcLength(contours2[j], true); // 윤곽선 전체 길이 계산, 열린곡선이므로 false
							vector<Point> result2;
							approxPolyDP(contours2[j], result2, perimeter * 0.04, true); // 다각형 근사, 윤곽선 배열 반환
							// 면적판단
							double area2 = contourArea(result2);

							if (debug && 16000 < area2) {
								cv::Mat temp;
								cvtColor(ROI2, temp, COLOR_GRAY2BGR);
								polylines(temp, result2, true, Scalar(0, 0, 255), 3);
								imshow("ROI2rect", temp);
								waitKey(0);
							}

							if (result2.size() >= 3 && 16000 < area2) { // = 800*800*0.1

								for (int k = 0; k < result2.size(); k++)
								{
									Point p1 = result2[(k + 0) % result2.size()];
									Point p0 = result2[(k + 1) % result2.size()];
									Point p2 = result2[(k + 2) % result2.size()];
									double angle = cosAngle(p1, p0, p2);

									if (debug) {
										cout << "angle: " << angle << endl;
										cout << p1 << p0 << p2 << endl;
										cout << norm(p1 - p0) << endl;
										cout << norm(p2 - p0) << endl << endl;
									}

									// arccos(0.17) = 80.21 degree
									// arccos(0.9) = 25.84 degree
									if (0.17 < angle && angle < 0.9 && p0.y > p1.y && p0.y > p2.y) {
										if (norm(p1 - p0) > 100 && norm(p0 - p2) > 100) {
											flag = false;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		if (flag != true) break;
		cv::rotate(gray, gray, ROTATE_90_CLOCKWISE);
	}

	// Mark detection 성공여부 확인
	if (flag != true) {
		for (int i = 0; i < rotate; i++)
			cv::rotate(src, src, ROTATE_90_CLOCKWISE);
	}
	else cout << "Mark was not detected!" << endl;

	return src;
}

cv::Mat getHist(cv::Mat image) {
	int nVals = 256;
	float range[] = { 0, 256.0 };
	const float* histRange = { range };
	cv::Mat hist;
	cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange);
	return hist;
}

vector<Point> Square(Mat src) {

	// 3 채널 분할
	Mat Split[3];
	cv::split(src, Split);
	vector<Point> square;

	if (src.channels() < 3) {
		cout << "error: src.channels() < 3" << endl;
	}
	else {
		double maxArea = 0;
		for (int channel = 1; channel < 3; channel++)
		{
			const int N = 10;

			int i;
			for (i = 0; i < N; i++)
			{
				Mat binary;
				threshold(Split[channel], binary, i * 255 / N, 255, cv::THRESH_BINARY);

				findBiggestSquare(binary, square, maxArea);
			}
		}
		return square;
	}
}

void findBiggestSquare(Mat src, vector<Point>& square, double& maxArea) {


	double max = src.cols * src.rows * 0.8;
	double min = src.cols * src.rows * 0.1;

	vector<vector<Point>> contours;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(
		src,
		contours,
		hierarchy,
		RETR_LIST,
		CHAIN_APPROX_TC89_KCOS
	);

	for (int j = 0; j < contours.size(); j++)
	{
		double perimeter = arcLength(contours[j], true); // 윤곽선 전체 길이 계산, 닫힌곡선이므로 true
		vector<Point> result;
		approxPolyDP(contours[j], result, perimeter * 0.03, true); // 다각형 근사, 윤곽선 배열 반환

		double area = contourArea(result); // 면적판단
		bool convex = isContourConvex(result); // 다각형 형태인지 볼록성 판단


		if (debug && result.size() == 4 && min < area && area < max && true) {
			cout << "result.size(): " << result.size() << endl;
			cout << "convex: " << convex << endl;
			cout << "max: " << max << endl;
			cout << "min: " << min << endl;
			cout << "area: " << area << endl;
			cout << "maxArea: " << maxArea << endl << endl;
			cv::Mat t;
			cvtColor(src, t, COLOR_GRAY2BGR);
			imshow("tdraw", DrawSquare(t, result));
			waitKey(0);
		}

		if (result.size() == 4 && min < area && area < max && convex) {
			int flag = 0;
			double cos = 0;
			for (int k = 0; k < 4; k++)
			{
				Point p1 = result[(k + 0) % 4];
				Point p0 = result[(k + 1) % 4];
				Point p2 = result[(k + 2) % 4];
				double t = cosAngle(p1, p0, p2);
				// t가 제일 큰, 즉 angle이 제일 작은 값을 찾아서 cos에 저장
				cos = cos > t ? cos : t;
				
				const int d = 10;
				if (	//세로
					(p1.x <= d && p0.x <= d)
					|| (src.cols - 4 <= p1.x && src.cols - 4 <= p0.x) ||	
						// 가로
					(p1.y <= d && p0.y <= d)
					|| (src.rows - 4 <= p1.y && src.rows - 4 <= p0.y)
					) {
					cout << "hi!"<< endl;

					Mat tmp = Mat::zeros(src.rows, src.cols, src.type());
					line(tmp, p1, p0, Scalar(255));
					if (sum(tmp & src).val[0] / sum(tmp).val[0] < 80.0){
						cout << "bye!" << endl;
						flag = -1;
						break;
					}
				}
			}

			// cos < 0.2
			// arccos(cos) > 78.46 degree
			// then isSquare = true
			if (flag != -1 && cos < 0.2 && maxArea < area) {
				square = result;
				maxArea = area;
			}
		}
	}
}

double cosAngle(Point pt1, Point pt0, Point pt2) {
	double ux = pt1.x - pt0.x;
	double uy = pt1.y - pt0.y;
	double vx = pt2.x - pt0.x;
	double vy = pt2.y - pt0.y;

	double uLen = sqrt(ux * ux + uy * uy);
	double vLen = sqrt(vx * vx + vy * vy);

	double dotProduct = ux * vx + uy * vy;

	// |u|*|v|*cos(angle) = dot(u , b) 공식을 사용
	return dotProduct / (uLen * vLen);
}

Mat DrawSquare(Mat src, vector<Point>& square)
{
	Mat drawsquare = src.clone();
	polylines(drawsquare, square, true, Scalar(255, 0, 255), 3);
	return drawsquare;
}

Mat getMarker(Mat src, vector<Point>& square)
{
	Mat marker(cv::Size(src.cols, src.rows), CV_8UC1, Scalar(0));

	// 테두리
	rectangle(marker,
		Point(0, 0),
		Point(src.cols, src.rows),
		Scalar(255),
		3
	);

	polylines(marker, square, true, Scalar(150), 3);

	// 무게중심 (cX,cY)
	Moments moments = cv::moments(square);
	Point2d m = { moments.m10 / moments.m00, moments.m01 / moments.m00 };
	const double rectSize = 100.0;
	rectangle(marker,
		m - Point2d(rectSize, rectSize),
		m + Point2d(rectSize, rectSize),
		Scalar(50),
		3
	);

	return marker;
}

Mat PerspectiveTransform(Mat src, vector<Point>& square)
{
	int f = 50000, s = 50000;
	int fi = -1, si = -1;
	for (int i = 0; i < 4; i++) {
		if (square[i].y < s) {
			if (square[i].y < f) {
				s = f;
				si = fi;
				f = square[i].y;
				fi = i;
			}
			else {
				s = square[i].y;
				si = i;
			}
		}
	}

	if (square[fi].x < square[si].x) {
		s = fi;

		if (fi < si) si = si - fi;
		else si = 4 - (fi - si);
	}
	else {
		s = si;

		if (si < fi) si = fi - si;
		else si = 4 - (si - fi);
	}

	// s :  왼쪽 위 꼭짓점 인덱스
	// si : 인덱스 -> 시계방향 or 반시계 방향? 
	Point2f src_pts[4];
	for (int i = 0; i < 4; i++) {
		src_pts[i] = Point2f(square[(s + i * si) % 4].x, square[(s + i * si) % 4].y);
	}

	double w = cv::norm(src_pts[0] - src_pts[1]);	// 가로 길이
	double h = cv::norm(src_pts[1] - src_pts[2]);	// 세로 길이

	int postitW = 800;
	int postitH = 800;

	// 가로 세로 길이가 많이 차이난다면 세로를 400으로 줄임
	if (abs(h - w) > (w > h ? w : h) * 0.4) {
		h > w ? postitW = 400 : postitH = 400;
	}

	Point2f dst_pts[4] =
	{
		Point2f(0, 0),
		Point2f(postitW, 0),
		Point2f(postitW, postitH),
		Point2f(0, postitH)
	};

	Mat dst;
	Mat matrix = cv::getPerspectiveTransform(src_pts, dst_pts);
	warpPerspective(src, dst, matrix, Size(postitW, postitH));
	return dst;
}