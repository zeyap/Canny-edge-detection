#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;
#define DELAY_CAPTION 1500
#define DELAY_DISPLAY 100
#define KERNEL_LENGTH 7
#define HIGH_THRESHOLD_PERCENTAGE 0.15f
#define PI 3.14f
Mat src; 
Mat dst; 
char window_name[] = "Canny Edge Detector Demo";
int display_caption(const char* caption, Mat dst);
int display_dst(int delay, Mat dst);
int display_caption(const char* caption);
int display_dst(int delay);

struct Gradient {
	float* M;
	float* theta;
	short* zeta;
	float* N;
};

Gradient gradient_sobel(Mat src, Mat &dst);
void nonMaxSuppression(Mat src, Mat &dst, Gradient &g);
void Sector(Gradient &g);
void thresholding(Mat src, Mat &dst, uchar t);
void estimateThreshold(Mat src, uchar* high, uchar* low);
void doubleThresholding(Mat src, Mat &dst, uchar tl, uchar th);
void hysteresis(Mat &dst, Mat ldst);
int * getNeigborIndex(Gradient &g, int i, int j);
bool findBreakpoint(int i, int j);
int trace(Mat* ldst, int i, int j, int neighbor,int* status);

int h, w;

int main(int argc, char ** argv)
{
	namedWindow(window_name, WINDOW_AUTOSIZE);
	const char* filename = argc >= 2 ? argv[1] : "gears.jpg";
	src = imread(filename, IMREAD_COLOR);
	h = src.size[0];
	w = src.size[1];
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Usage: ./Smoothing [image_name -- default ..lena.bmp] \n");
		return -1;
	}
	if (display_caption("Original Image") != 0) { return 0; }
	dst = src.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	if (display_caption("Grey Scale") != 0) { return 0; }
	cvtColor(src,dst,CV_BGR2GRAY);
	src = dst.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	if (display_caption("Gaussian Blur") != 0) { return 0; }
	GaussianBlur(src, dst, Size(KERNEL_LENGTH, KERNEL_LENGTH), 0, 0);
	src = dst.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }
	
	if (display_caption("Gradient") != 0) { return 0; }
	Gradient g = gradient_sobel(src, dst);
	src = dst.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }
	
	if (display_caption("Non-Max Suppression") != 0) { return 0; }
	nonMaxSuppression(src, dst,g);
	src = dst.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }
	
	Mat newDst = src.clone();

	uchar thigh = 0;
	uchar tlow = 0;
	estimateThreshold(src, &thigh, &tlow);

	if (display_caption("Higher Thresholding") != 0) { return 0; }
	thresholding(src, newDst, thigh);
	if (display_dst(DELAY_CAPTION, newDst) != 0) { return 0; }

	if (display_caption("Double Thresholding") != 0) { return 0; }
	doubleThresholding(src, dst, thigh, tlow);
	src = dst.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	display_caption("Done!");
	return 0;
}
int display_caption(const char* caption, Mat dst)
{
	dst = Mat::zeros(src.size(), src.type());
	putText(dst, caption,
		Point(src.cols / 4, src.rows / 2),
		FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
	return display_dst(DELAY_CAPTION, dst);
}
int display_dst(int delay, Mat dst)
{
	imshow(window_name, dst);
	int c = waitKey(delay);
	if (c >= 0) { return -1; }
	return 0;
}
int display_caption(const char* caption)
{
	return display_caption(caption,dst);
}
int display_dst(int delay)
{
	return display_dst(delay,dst);
}

Gradient gradient_sobel(Mat src, Mat &dst) {

	float gradient[2];

	Gradient g;
	g.M = new float[src.size[0] * src.size[1]];
	g.theta = new float[src.size[0] * src.size[1]];
	
	for (int i = 1; i < h-1;i++) {
		for (int j = 1; j < w-1; j++) {
			gradient[0]=
				-src.at<uchar>(i-1,j-1)
				+src.at<uchar>(i-1,j+1)
				-2* src.at<uchar>(i, j-1)
				+2* src.at<uchar>(i, j + 1)
				- src.at<uchar>(i+1, j-1) 
				+ src.at<uchar>(i + 1, j+1);
			gradient[1] = 
				src.at<uchar>(i - 1, j - 1)
				+2* src.at<uchar>(i - 1, j)
				+ src.at<uchar>(i - 1, j + 1)
				- src.at<uchar>(i+1, j - 1)
				-2* src.at<uchar>(i + 1, j)
				- src.at<uchar>(i + 1, j + 1);
			g.M[i*w + j] = sqrt(gradient[0]*gradient[0]+gradient[1]* gradient[1]);
			g.theta[i*w + j] = atan2(gradient[1],gradient[0]);
			dst.at<uchar>(i, j) = (uchar)g.M[i*w + j];
		}
	}
	return g;
}

void nonMaxSuppression(Mat src, Mat &dst, Gradient &g) {
	Sector(g);
	g.N = new float[w*h];
	int* index;
	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			index=getNeigborIndex(g,i,j);
			if (g.M[i*w + j] < g.M[index[0]] || g.M[i*w + j] < g.M[index[1]]) {
				g.N[i*w + j] = 0;
			}
			else {
				g.N[i*w + j] = g.M[i*w + j];
			}
			dst.at<uchar>(i, j) = (uchar)g.N[i*w + j];
		}
	}

}

int * getNeigborIndex(Gradient &g, int i, int j) {
	int index[2];
	switch (g.zeta[i*w + j]) {
	case 0:
		index[0] = i*w + j - 1;
		index[1] = i*w + j + 1;
		break;
	case 1:
		index[0] = (i - 1)*w + j + 1;
		index[1] = (i + 1)*w + j - 1;
		break;
	case 2:
		index[0] = (i - 1)*w + j;
		index[1] = (i + 1)*w + j;
		break;
	case 3:
		index[0] = (i - 1)*w + j - 1;
		index[1] = (i + 1)*w + j + 1;
		break;
	}
	return index;
}

void Sector(Gradient &g) {
	g.zeta=new short[w*h];
	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			float t = g.theta[i*w + j] + PI / 8;
			if ((t >= 0 && t < PI / 4) || (t >= -PI && t < -PI * 3 / 4)) {
				g.zeta[i*w + j] = 0;//vertical edge
			}
			else if ((t >= PI/4 && t < PI / 2) || (t >= -PI*3/4 && t < -PI / 2)) {
				g.zeta[i*w + j] = 1;//+45 edge
			}
			else if ((t >= PI / 2 && t < PI *3/ 4) || (t >= -PI / 2 && t < -PI / 4)) {
				g.zeta[i*w + j] = 2;//horizontal edge
			}
			else {
				g.zeta[i*w + j] = 3;//-45 edge
			}
		}
	}
}

void thresholding(Mat src, Mat &dst, uchar t) {

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) < t) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

void doubleThresholding(Mat src, Mat &dst, uchar th, uchar tl) {
	Mat ldst;
	ldst=src.clone();
	thresholding(src, dst, th);
	thresholding(src, ldst, tl);
	hysteresis(dst,ldst);
}

void estimateThreshold(Mat src, uchar* high, uchar* low) {
	int histogram[256];
	uchar max = 0;
	int totalPixelNum = 0;
	for (int i = 0; i < 256; i++) {
		histogram[i] = 0;
	}
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int temp = src.at<uchar>(i, j);
			if (temp > max) {
				max = temp;
			}
			histogram[temp]++;
		}
	}
	for (int i = 0; i < 255; i++) {
		totalPixelNum+=histogram[i];
	}
	int pixelNum = (totalPixelNum - histogram[0])*HIGH_THRESHOLD_PERCENTAGE;
	int highCutoff = max;
	int beyondCutoffSum=0;
	do {
		beyondCutoffSum += histogram[highCutoff];
		highCutoff--;
	} while (beyondCutoffSum < pixelNum);
	*high = highCutoff;
	*low = highCutoff*0.5;
}

void hysteresis(Mat &dst, Mat ldst) {
	int* status = new int[w*h];//0-low/zero 1-high 2-low&accepted
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (dst.at<uchar>(i, j) == 255) {
				status[i*w + j] = 1;
			}else{
				status[i*w + j] = 0;
			}
		}
	}
	
	for (int i = 1; i < h-1; i++) {
		for (int j = 1; j < w-1; j++) {
			if (findBreakpoint(i,j)) {//is a breakpoint
				int y,x;
				y = i;
				x = j;
				trace(&ldst,y,x,0, status);
			}
		}
	}

}

const int yOffset[] = {0,1,1,1,0,-1,-1,-1};
const int xOffset[] = {-1,-1,0,1,1,1,0,-1};

bool findBreakpoint(int i,int j) {
	uchar eightNeighbor[8];
	int sum = 0;
	for (int k = 0; k < 8; k++) {
		eightNeighbor[k] = dst.at<uchar>(i + yOffset[k], j + xOffset[k]);
		sum += eightNeighbor[k] / 255;
	}
	return sum == 1;
}

int trace(Mat* ldst, int i, int j, int neighbor, int* status) {
	uchar eightNeighborLow[8];
	for (int k = 0; k < 8; k++) {
		eightNeighborLow[k] = ldst->at<uchar>(i+yOffset[k], j + xOffset[k]);
	}

	while (1) {
		for (int k = 0; k < 8; k++) {
			int neighborIndex = (neighbor + k) % 8;
			if (eightNeighborLow[neighborIndex] != 0) {
				int y = i + yOffset[neighborIndex];
				int x = j + xOffset[neighborIndex];
				if (status[y*w + x] == 0) {
					//accept this point
					dst.at<uchar>(y, x) = 255;
					status[y*w + x] = 2;
					//update
					neighbor = (neighborIndex +7)%8;
					return trace(ldst, y, x, neighbor, status);
				}
				else {
					return 1;
				}
			}
			else {//falls below lower threshold
				if (k == 7) {
					return 0;
				}
			}
			
		}
	}
}

