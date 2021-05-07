// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <minwindef.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cmath>
#include <queue>
#include <utility> 
#include <random>
#include <climits>
#include <fstream>
#define pi 3.14159265358979323846

std::default_random_engine gen;
std::uniform_int_distribution<int> dgen(0, 255);

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void IncreaseGrayLevel(int factor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int neg = val + factor;
				if (neg > 255)
					neg = 255;
				if (neg < 0)
					neg = 0;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		// Print (in the console window) the processing time in [ms] 
		imshow("input image", src);
		imshow("Gray Lev image", dst);
		waitKey();
	}
}

void MultiplicativeGray(int factor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int neg = val * factor;
				if (neg > 255)
					neg = 255;
				if (neg < 0)
					neg = 0;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		// Print (in the console window) the processing time in [ms] 
		imshow("input image", src);
		imshow("Gray Lev image", dst);
		waitKey();
	}
}

void CreateColorImag()
{
	int height, width;
	height = 256;
	width = 256;
	unsigned char B ;
	unsigned char G ;
	unsigned char R ;

	Mat scr = Mat(height, width, CV_8UC3);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (j <= 125 && i <= 125) //zone 1 - white
			{
				 B = 255;
				 G = 255;
				 R = 255;

			}
			if (j > 125 && i <= 125)	//zone 2 - red
			{
				 B = 0;
				 G = 0;
				 R = 255;
			}
			if (j <= 125 && i > 125) //zone 3 -green
			{
				 B = 0;
				 G = 255;
				 R = 0;
			}
			if (j > 125 && i > 125)	//zone 4 - yellow
			{
				 B = 0;
				 G = 255;
				 R = 255;
			}
			scr.at<Vec3b>(i, j)[0] = B;
			scr.at<Vec3b>(i, j)[1] = G;
			scr.at<Vec3b>(i, j)[2] = R;
		}
		
	}
		imshow("image", scr);
		waitKey();
}

void MakeFloatMatrix()
{
	int x;
	float vals[9] = {1,0,0,0,4,0,0,2,1};
	
	Mat something = Mat(3, 3, CV_32FC1,vals); //data type is float alright
	//Mat negative = Mat(3, 3, CV_32FC1); // output
	Mat negative = something.inv();

	std::cout << negative << std::endl;
	scanf("%d",&x);
}

void Make3Channels()
{
	Mat_<Vec3b> src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		
		Mat_<uchar> DR(src.rows, src.cols);
		Mat_<Vec3b> DRC(src.rows, src.cols);

		Mat_<uchar> DB(src.rows, src.cols);
		Mat_<Vec3b> DBC(src.rows, src.cols);

		Mat_<uchar> DG(src.rows, src.cols);
		Mat_<Vec3b> DGC(src.rows, src.cols);

		for (int i = 0; i < src.rows; i++)	//Blue channel
		{
			for (int j = 0; j < src.cols; j++)
			{
				DBC(i, j)[0] = src(i, j)[0];
				DBC(i, j)[1] = 0;
				DBC(i, j)[2] = 0;
				DB(i, j) = src(i, j)[0];

			}
		}
		
		for (int i = 0; i < src.rows; i++)	//Green channel
		{
			for (int j = 0; j < src.cols; j++)
			{
				DGC(i, j)[1] = src(i, j)[1];
				DGC(i, j)[0] = 0;
				DGC(i, j)[2] = 0;
				DG(i, j) = src(i, j)[1];

			}
		}

		for (int i = 0; i < src.rows; i++)	//Red channel
		{
			for (int j = 0; j < src.cols; j++)
			{
				DRC(i, j)[2] = src(i, j)[2];
				DRC(i, j)[1] = 0;
				DRC(i, j)[0] = 0;
				DR(i, j) = src(i, j)[2];

			}
		}
		
		imshow("input image", src);
		imshow("RED", DR);
		imshow("ColorR", DRC);
		imshow("BLOO", DB);
		imshow("ColorB", DBC);
		imshow("GREEN", DG);
		imshow("ColorG", DGC);
		waitKey();
	}
}

void ToGrayScale()
{
	Mat_<Vec3b> src;
	
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);

		Mat_<uchar> dst(src.rows,src.cols);
		
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				dst(i, j) = (src(i, j)[0] + src(i, j)[1] + src(i, j)[2])/3;
			}
		}
		imshow("original", src);
		imshow("dst",dst);
		waitKey();
	}
}

void ToBinary()
{
	int x;
	Mat_<uchar> src;
	
	printf("choose a threshold: ");
	scanf("%d", &x);

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname,IMREAD_GRAYSCALE);
		
		Mat_<uchar> dst(src.rows, src.cols);
		
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src(i, j) >= x)
					dst(i, j) = 255; //white
				else
					dst(i, j) = 0;	//black
			}
		}
			imshow("Original", src);
			imshow("Destination", dst);;
			waitKey();
	}
}

float maxf(float x, float y, float z)
{
	return max(max(x, y), z);
}

float minf(float x, float y, float z)
{
	return min(min(x, y), z);
}

void toHSV()
{
	Mat_<Vec3b> src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		
		Mat_<uchar> Hue(src.rows, src.cols);
		Mat_<uchar> Sat(src.rows, src.cols);
		Mat_<uchar> Val(src.rows, src.cols);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float r = src(i, j)[2]/255.0;
				float g = src(i, j)[1]/255.0;
				float b = src(i, j)[0]/255.0;

				float M = maxf(r, g, b);
				float m = minf(r, g, b);

				float h, s, v;

				float C = M - m;
				//Value 
				v=M;
	
				//Saturation
				if (v != 0)
				{
					s=C/v;
				}
				else
					s = 0;

				//Hue
				if (C != 0)
				{
					if (M == r)
					{
						h = (60 * (g - b) / C);
					}
					if (M == g)
					{
						h = (120 + 60 * (b-r) / C);
					}
					if (M == b)
					{
						h = (240 + 60 * (r-g) / C);
					}

				}
				else
					h = 0;

				if (h < 0)
					h = h + 360;

				Hue(i, j) = h * 255 / 360;
				Sat(i, j) = s * 255;
				Val(i, j) = v * 255;


			}
		}

		imshow("original",src);
		imshow("Hue", Hue);
		imshow("Saturation", Sat);
		imshow("Value",Val);
		waitKey();

	}
}

void isInside()
{
	Mat_<Vec3b> img;

	int i, j;

	printf("please give us a coordinate\n x= ");
	scanf("%d",&i);
	printf(" y= ");
	scanf("%d", &j);

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		img = imread(fname);

		if (((i < img.rows) && (j < img.cols)) && ((i >= 0) && (j >= 0)))
		{
			printf("is inside\n");
		}
		else
		{
			printf("is outside\n");
		}
	}
}

void makeHistoGR(Mat_<uchar> src)
{
	int histogram[256];

	for (int i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			histogram[src(i, j)]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		printf("%d ", histogram[i]);
	}

	printf("\n\n");

	showHistogram("Histogram", histogram, 255, 255);
	waitKey(0);
}

void multiLevelThreshold(float pdf[256], int histograma[256], int height, int width, Mat_<uchar> dst)
{
	printf("\n\n");
	const int WH = 5;
	const float TH = 0.0003;
	int roi_size = 2 * WH + 1;
	std::vector<int> maximum_values;

	for (int k = 0 + WH; k <= 255 - WH; k++)
	{
		float v = 0;
		bool greaterThan = true;
		for (int j = k - WH; j <= k + WH; j++)
		{
			v += pdf[j];
			if (pdf[k] < pdf[j])
				greaterThan = false;
		}
		v /= roi_size;
		if (pdf[k] > v + TH && greaterThan)
			maximum_values.push_back(k);
	}
	maximum_values.insert(maximum_values.begin(), 0);
	maximum_values.insert(maximum_values.end(), 255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int v = (int)dst(i, j);
			int closest = 9999999;
			for (int l = 1; l < maximum_values.size(); l++)
			{
				if (v <= maximum_values[l])
				{
					closest = maximum_values[l] - v < v - maximum_values[l - 1] ? maximum_values[l] : maximum_values[l - 1];
					break;
				}
			}
			dst(i, j) = closest;
		}
	}

	imshow("Grey Threshold Image", dst);
	makeHistoGR(dst);
	waitKey();
}

void computePDF(int histogram[256], int height, int width,Mat_<uchar> dst)
{
	float PDF[256];
	int M = height*width;

	for (int i = 0; i < 256; i++)
	{
		
		PDF[i] = (float) histogram[i] / M;
		printf("%f ",PDF[i]);
	}
	printf("\n");
	multiLevelThreshold(PDF, histogram, height, width,dst);

}

void makeHisto()
{
	Mat_<uchar> src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE); //we have our greyscale image

		int histogram[256];
		for (int i = 0; i < 256; i++)
		{
			histogram[i] = 0;
		}

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				histogram[src(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			printf("%d ", histogram[i]);
		}

		printf("\n\n");

		computePDF(histogram, src.rows,src.cols,src);

		showHistogram("Histogram", histogram, 255, 255);
		waitKey(0);

	}
}

void makeHistoWithBins(int m)
{
	Mat_<uchar> img;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		img = imread(fname, IMREAD_GRAYSCALE); //we have our greyscale image

		int binsize = 256 / m;
		
		int *histogram = new int[m];

		for (int i = 0; i < m; i++)
		{
			histogram[i] = 0;
		}

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				int val = (int)img(i, j);
				histogram[val/binsize]++;
			}
		}

		for (int i = 0; i < m; i++)
		{
			printf("%d ",histogram[i]);
		}

		showHistogram("Histogram", histogram, m, img.cols);
		waitKey(0);
	}
}
float* calc_histogram_fdp(Mat_<uchar> src)
{
	int histogram[256];
	float pdf[256];
	int M = src.cols*src.rows;

	for (int i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			histogram[src(i, j)]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		pdf[i] = (float)histogram[i] / M;
	}
	printf("\n");
	
	return pdf;

}

void floyd_steinberg_dithering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = src.clone();

		int height = src.rows;
		int width = src.cols;
		const int WH = 5;
		const float TH = 0.0003;
		int roi_size = 2 * WH + 1;

		float *pdf = calc_histogram_fdp(src);

		std::vector<int> maximum_values;

		for (int k = 0 + WH; k <= 255 - WH; k++)
		{
			float v = 0;
			bool greaterThan = true;

			for (int j = k - WH; j <= k + WH; j++)
			{
				v += pdf[j];
				if (pdf[k] < pdf[j])
					greaterThan = false;
			}
			v /= roi_size;
			if (pdf[k] > v + TH && greaterThan)
				maximum_values.push_back(k);
		}
		maximum_values.insert(maximum_values.begin(), 0);
		maximum_values.insert(maximum_values.end(), 255);

		/*Second Step*/
		for (int i = 1; i <= height - 2; i++)
		{
			for (int j = 1; j <= width - 2; j++)
			{
				int v = (int)dst(i, j);
				int closest = 9999999;
				
				for (int l = 1; l < maximum_values.size(); l++)
				{
					if (v <= maximum_values[l])
					{
						closest = maximum_values[l] - v < v - maximum_values[l - 1] ? maximum_values[l] : maximum_values[l - 1];
						break;
					}
				}
				dst(i, j) = closest;

				double error = v - closest;
				dst(i, j + 1) = min(255, max(0, dst(i, j + 1) + (int)(7 * error / 16)));
				dst(i + 1, j - 1) = min(255, max(0, dst(i + 1, j - 1) + (int)(3 * error / 16)));
				dst(i + 1, j) = min(255, max(0, dst(i + 1, j) + (int)(5 * error / 16)));
				dst(i + 1, j + 1) = min(255, max(0, dst(i + 1, j + 1) + (int)(error / 16)));

			}
		}
		imshow("original", src);
		imshow("Floyd Steinber Image", dst);
		waitKey();
	}
}

int is_perimeter_point(Mat sourceImage, int x, int y, Vec3b backgroundgColor)
{
	int neigh[] = { -1,0,1 };

	for (int dy = 0; dy < 3; dy++)
		for (int dx = 0; dx < 3; dx++)
		{
			int nx = x + neigh[dx];
			int ny = y + neigh[dy];

			//test if inside image boundries
			if (ny >= sourceImage.rows || nx >= sourceImage.cols || ny < 0 || nx < 0)
				continue;

			//if it has one neighbour which is background => perimter point
			if (sourceImage.at<Vec3b>(ny, nx) == backgroundgColor)
				return 1;

		}

	return 0;
}

void process_object_handler(int event, int ex, int ey, int flags, void* param)
{
	
	if (event != EVENT_LBUTTONDOWN) return;

	Mat_<Vec3b> src = *(Mat*)param;

	Mat_<Vec3b> object = Mat(src.rows, src.cols, CV_8UC3);
	Mat_<Vec3b> projection = Mat(src.rows, src.cols, CV_8UC3);

	int height = src.rows;
	int width = src.cols;
	Vec3b back_color = src(0, 0);
	Vec3b object_color = src(ey, ex);

	int min_x = width, max_x = 0, min_y = height, max_y = 0;
	int area = 0; int perimeter_value = 0;
	int cx = 0, cy = 0;

	int *projection_x = new int[width];
	int *projection_y = new int[height];
	memset(projection_y, 0, sizeof(int) * height);
	memset(projection_x, 0, sizeof(int) * width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) != object_color)
				continue;

			if (x > max_x)
				max_x = x;
			if (x < min_x)
				min_x = x;

			if (y > max_y)
				max_y = y;

			if (y < min_y)
				min_y = y;

			object(y, x) = object_color;

			projection_x[x]++;
			projection_y[y]++;

			cx = cx + x;
			cy = cy + y;

			area++; 

			if (is_perimeter_point(src, x, y, back_color)) 
			{
				perimeter_value++;
				object(y, x) = Vec3b(0, 0, 0);
			}

		}
	}

	cx = cx/area;
	cy = cy/area;
	circle(object, Point(cx, cy), 3, Scalar(0, 0, 0), -1);

	int num = 0;
	int nim = 0;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) != object_color)
				continue;

			num = num + (y - cy)*(x - cx);
			nim = nim + (x - cx)*(x - cx) - (y - cy)*(y - cy);

		}
	}

	double angle = atan2(2.0*num, nim) / 2.0;

	Point p1(cx, cy);
	Point p2(cx + 30 * cos(angle), cy + 30 * sin(angle));
	Point p3(cx - 30 * cos(angle), cy - 30 * sin(angle));

	line(object, p1, p2, Scalar(0, 0, 0));
	line(object, p1, p3, Scalar(0, 0, 0));

	double aspect_ratio = 1.0 * (max_x - min_x + 1) / (max_y - min_y + 1);
	double thinning_factor = 4.0 * PI * area / (perimeter_value * perimeter_value);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < projection_y[y]; x++)
		{
			projection(y, x) = Vec3b(0, 0, 0);
		}
	}

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < projection_x[x]; y++)
		{
			projection(y, x) = Vec3b(0, 0, 0);
		}
	}
	printf("\n");
	printf("Area: %d\n", area);
	printf("Perimiter: %d\n",perimeter_value);
	printf("Center: %d %d\n",cx,cy);
	printf("Angle: %f\n",angle*180/PI);
	printf("Aspect Ratio: %f\n",aspect_ratio);
	printf("Thinning factor: %f\n",thinning_factor);

	imshow("Object Image", object);
	imshow("Projection Image", projection);
	waitKey();
}

void process_objects()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname);

		imshow("Orig Image", src);

		setMouseCallback("Orig Image", process_object_handler, &src);

		waitKey();
		destroyAllWindows();
	}
}

//-------===================== L A B _ F I V E (5) ===================-------//

enum NeighboursType { EIGHT, FOUR, PREVIOUS };

std::vector<Point2i> getNeighbours(int x, int y, int width, int height, NeighboursType type) 
{
	std::vector<Point2i> points;
	int neighbours_size = (type == EIGHT) ? 8 : 4;

	int neighboursFourX[4] = { 1,0,-1,0 };
	int neighboursFourY[4] = { 0,1,0,-1 };

	int neighboursEightX[8] = { 1,1,0,-1,-1,-1,0 };
	int neighboursEightY[8] = { 0,1,1,1,0,-1,-1 };

	int neighboursPrevX[4] = { -1,-1,0,1 };
	int neighboursPrevY[4] = { 0,-1,-1,-1 };

	std::vector<int> neighboursX;
	std::vector<int> neighboursY;

	for (int i = 0; i< neighbours_size; i++)
	{
		switch (type)
		{
		case FOUR:
			neighboursX.push_back(neighboursFourX[i]);
			neighboursY.push_back(neighboursFourY[i]);
			break;
		case EIGHT:
			neighboursX.push_back(neighboursEightX[i]);
			neighboursY.push_back(neighboursEightY[i]);
		case PREVIOUS:
			neighboursX.push_back(neighboursPrevX[i]);
			neighboursY.push_back(neighboursPrevY[i]);
		}
	}

	for (int k = 0; k < neighbours_size; k++)
	{
		int nx = x + neighboursX[k];
		int ny = y + neighboursY[k];

		if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

		points.emplace_back(nx, ny);
	}

	return points;
}

Mat color_labeled_image(Mat_<short> labels, int height, int width, unsigned int nrLabels)
{
	Mat_<Vec3b> colored_labels = Mat::zeros(cv::Size(width, height), CV_8UC3);

	std::vector<Vec3b> label_list;
	label_list.push_back(Vec3b(255, 255, 255));//tfw white background


	for (unsigned int i = 1; i < nrLabels + 1; i++)
		label_list.push_back(Vec3b(dgen(gen), dgen(gen), dgen(gen)));

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int label = labels(i, j);
			colored_labels(i, j) = label_list[label];
		}
	}
	return colored_labels;
}

void breadth_first_labeling()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<short> labels = Mat::zeros(cv::Size(width, height), CV_16SC1);

		short current_label = 0;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
				if (src(y, x) == 0 && labels(y, x) == 0) 
				{

					current_label++;
					std::queue<Point2i> queue;

					labels(y, x) = current_label;
					queue.push(Point2i(x, y));

					while (!queue.empty())
					{
						Point2i q = queue.front();
						queue.pop();

						std::vector<Point2i> neighbours = getNeighbours(q.x, q.y, width, height, EIGHT);

						for (int k = 0; k < neighbours.size(); k++) 
						{
							Point2i n = neighbours[k];
							if (src(n.y, n.x) == 0 && labels(n.y, n.x) == 0)
							{
								labels(n.y, n.x) = current_label;
								queue.push(Point2i(n.x, n.y));
							}
						}
					}

				}
		}
		Mat colored_labels = color_labeled_image(labels, height, width, current_label);

		imshow("Original Image", src);
		imshow("Labeled Image", colored_labels);
		waitKey();
	}
}

short vector_minimum(std::vector<short> data)
{
	short minimum = data[0];
	for (auto d : data)
		if (d < minimum)
			minimum = d;
	return minimum;
}

void equivalence_class_labeling()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{

		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<short> labels = Mat::zeros(cv::Size(width, height), CV_16SC1);
		short current_label = 0;
		std::vector<std::vector<int>> edges;

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				if (src(y, x) == 0 && labels(y, x) == 0) 
				{
					std::vector<short> L;

					auto neighbours = getNeighbours(x, y, width, height, PREVIOUS);
					for (auto n : neighbours)
					{
						short clabel = labels(n.y, n.x);
						if (clabel  > 0)
							L.push_back(clabel);
					}

					if (L.size() == 0) {
						current_label++;
						labels(y, x) = current_label;
						edges.resize(current_label + 1);
					}
					else {
						short val_min = vector_minimum(L);
						labels(y, x) = val_min;
						for (auto d : L)
							if (d != val_min)
							{
								edges[val_min].push_back(d);
								edges[d].push_back(val_min);
							}
					}
				}

		short new_label = 0;
		short *new_label_list = new short[current_label + 1];
		memset(new_label_list, 0, sizeof(short)*(current_label + 1));
		for (int k = 1; k <= current_label; k++)
			if (new_label_list[k] == 0)
			{
				new_label++;
				std::queue<short> queue;
				new_label_list[k] = new_label;
				queue.push(k);
				while (!queue.empty())
				{
					short x = queue.front();
					queue.pop();
					for (auto y : edges[x])
						if (new_label_list[y] == 0)
						{
							new_label_list[y] = new_label;
							queue.push(y);
						}
				}
			}
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				labels.at<short>(y, x) = new_label_list[labels.at<short>(y, x)];
		Mat colored_labels = color_labeled_image(labels, height, width, current_label);


		imshow("Original Image", src);
		imshow("Labeled Image", colored_labels);
		waitKey();
	}
}

std::vector<Point3i> getNeighboursWithDirection(int x, int y, int width, int height, NeighboursType type, bool willSort = false, int dir = 5) 
{
	std::vector<Point3i> points;
	int neighbours_size = (type == EIGHT) ? 8 : 4;

	int neighboursFourX[4] = { 1,0,-1,0 };
	int neighboursFourY[4] = { 0,-1,0,1 };

	int neighboursEightX[8] = { 1, 1, 0,-1,-1,-1,0,1 };
	int neighboursEightY[8] = { 0,-1,-1,-1, 0, 1,1,1 };

	std::vector<int> neighboursX;
	std::vector<int> neighboursY;
	std::vector<int> neighboursDir;

	for (int i = 0; i < neighbours_size; i++)
	{
		switch (type)
		{
		case FOUR:
			neighboursX.push_back(neighboursFourX[i]);
			neighboursY.push_back(neighboursFourY[i]);
			break;
		case EIGHT:
			neighboursX.push_back(neighboursEightX[i]);
			neighboursY.push_back(neighboursEightY[i]);
		default:
			break;
		}
		neighboursDir.push_back(i);
	}

	//USED FOR CANNY EDGE DETECTION
	if (willSort && type != PREVIOUS)
	{
		std::vector<int> neighboursX_clone(neighboursX);
		std::vector<int> neighboursY_clone(neighboursY);
		std::vector<int> neighboursDir_clone(neighboursDir);

		neighboursX.clear();
		neighboursY.clear();
		neighboursDir.clear();

		int start = 0;
		switch (type)
		{
		case FOUR:
			dir = (dir + 3) % 4;
			break;
		case EIGHT:
			dir = (dir + ((dir % 2 == 0) ? 7 : 6)) % 8; //If odd do + 6 then mod 8/////////// else if even do +7 then mod 8
			break;
		default:
			dir = 7;
			break;
		}
		for (int i = (dir); i < neighbours_size; i++) {
			neighboursX.push_back(neighboursX_clone[i]);
			neighboursY.push_back(neighboursY_clone[i]);
			neighboursDir.push_back(neighboursDir_clone[i]);
		}
		for (int i = 0; i < (dir); i++) {
			neighboursX.push_back(neighboursX_clone[i]);
			neighboursY.push_back(neighboursY_clone[i]);
			neighboursDir.push_back(neighboursDir_clone[i]);
		}
	}

	for (int k = 0; k < neighbours_size; k++)
	{
		int nx = x + neighboursX[k];
		int ny = y + neighboursY[k];

		if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

		points.emplace_back(nx, ny, neighboursDir[k]);
	}

	return points;
}

bool contour_control(Point2i current, Point2i prev, Point2i start2, Point2i start1, bool first)
{
	if (first) return first;

	return current != start2 && prev != start1; //this returns 1 if it's the first pixel we encounter else, it returns 1 if we did a full contour scan aka returned to original point
}

void countours(NeighboursType type)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);

		int height = img.rows;
		int width = img.cols;

		Mat_<uchar> src = Mat(Size(width, height), CV_8UC1, Scalar(255));
		threshold(img, src, 128, 255, 0);
		imshow("Original Image", src);

		Mat_<uchar> dst = Mat(Size(width, height), CV_8UC1, Scalar(255));

		Point2i start1(-1, -1);
		Point2i start2(-1, -1);
		Point2i nullPoint(-1, -1);

		int dir = (type == EIGHT) ? 5 : 4;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++)
				if (src(y, x) == 0) // - reach first black pixel in the image
				{
					start1 = Point2i(x, y);
					y = height;
					x = width;
				}
		}

		std::vector<Point3i> contour;

		Point2i current = start1;
		Point2i prev(0, 0);

		bool first = true;
		do {
			if (start2 != nullPoint)
				first = false;

			contour.push_back(Point3i(current.x, current.y, dir));
			std::vector<Point3i> neighbours = getNeighboursWithDirection(current.x, current.y, width, height, type, true, dir);
			for (auto n : neighbours)
				if (src(n.y, n.x) == 0) 
				{
					prev = current;
					current = Point2i(n.x, n.y);

					if (start2 == nullPoint)
					{
						start2 = current;
					}

					dir = n.z;
					break;
				}
		} while (contour_control(current, prev, start2, start1, first));

		std::vector<int> derivates; 
		int d;
		
		for (int i = 1; i<contour.size(); i++)
		{
			d = contour[i].z - contour[i - 1].z;
			d = (d < 0) ? d + 8 : d;
			derivates.push_back(d);
		}
		if (contour.size() > 0)
		{
			d = contour[contour.size() - 1].z - contour[0].z;
			d = (d < 0) ? d + 8 : d;
			derivates.push_back(d);
		}

		for (auto p : contour)
		{
			dst(p.y, p.x) = 0;
		}
		
		printf("\n\nThere are %d countour points\n They are: ",contour.size()-2);

		for (int i = 1; i < contour.size()-1; i++)
		{
			printf("%d ",contour[i].z);
		}

		printf("\n\nThere are %d derivatives points\n They are: ", derivates.size() - 2);
		for (int i=1;i<derivates.size()-1;i++)
		{
			printf("%d ",derivates[i]);
		}

		imshow("Original Image", src);
		imshow("Contour Image", dst);
		waitKey();
	}
}

void reconstruct_from_text()
{
	int neighboursEightX[8] = { 1,1,0,-1,-1,-1,0,1 };
	int neighboursEightY[8] = { 0,1,1,1,0,-1,-1,-1 };

	std::ifstream inputFile;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		inputFile.open("C:\\Users\\dbutu\\Desktop\\Poli Stuff\\Image Processing\\OpenCVApplication-VS2017_OCV340_basic\\OpenCVApplication-VS2017_OCV340_basic\\Images\\reconstruct.txt");
		Mat_<uchar> img = imread(fname,IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;
		Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
		int sx, sy, n_size;
		inputFile >> sy;
		inputFile >> sx;
		inputFile >> n_size;

		for (int i = 0; i<n_size; i++)
		{
			int dir;
			inputFile >> dir;

			sx += neighboursEightX[dir];
			sy += neighboursEightY[dir];

			if (sy < height || sx < width || sy >= 0 || sx >= 0)
				dst(height - sy, sx) = 0;
		}

		imshow("Reconstructed Image", dst);
		waitKey();
	}
}

std::vector<Point2i> structuralNeighbours(int width, int height, Mat struc, Point2i localPoint, uchar ignoreColor)
{
	std::vector<Point2i> neighbours;
	int heightStruct = struc.rows;
	int widthStruct = struc.cols;

	Point2i strucCenter(widthStruct / 2, heightStruct / 2);

	for (int y = 0; y < heightStruct; y++)
	{
		for (int x = 0; x < widthStruct; x++)
		{
			if (struc.at<uchar>(y, x) != ignoreColor) 
			{
				int dx = localPoint.x - (strucCenter.x - x);
				int dy = localPoint.y - (strucCenter.y - y);

				if (dx < 0 || dy < 0 || dx >= width || dy >= height) continue;

				neighbours.push_back(Point2i(dx, dy));
			}
		}
	}
	return neighbours;
}

Mat matrix_dillate(Mat_<uchar> srcImg, Mat struc, uchar backgroundColor)
{
	int height = srcImg.rows;
	int width = srcImg.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			auto neighbours = structuralNeighbours(width, height, struc, Point2i(x, y), backgroundColor);

			int color = 255;
			for (auto n : neighbours)
			{
				color = min(srcImg(n.y, n.x), color);
			}

			dst(y, x) = color;
		}
	}
	return dst;
}

Mat matrix_erode(Mat_<uchar> srcImg, Mat struc, uchar backgroundColor)
{
	int height = srcImg.rows;
	int width = srcImg.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			auto neighbours = structuralNeighbours(width, height, struc, Point2i(x, y), backgroundColor);

			int color = 0;
			for (auto n : neighbours)
			{
				color = max(srcImg(n.y, n.x), color);
			}

			dst(y, x) = color;
		}
	}
	return dst;
}

Mat matrix_difference(Mat_<uchar> src, Mat_<uchar> cmp, uchar backgroundColor)
{
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) != backgroundColor && cmp(y, x) == backgroundColor)
			{
				dst(y, x) = 0;
			}
		}
	}
	return dst;
}

Mat matrix_and(Mat_<uchar> src, Mat_<uchar> cmp, uchar backgroundColor)
{
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) == cmp(y, x) && cmp(y, x) != backgroundColor)
			{
				dst(y, x) = 0;
			}
		}
	}
	return dst;
}

Mat matrix_union(Mat_<uchar> src, Mat_<uchar> cmp, uchar backgroundColor)
{
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) != backgroundColor || cmp(y, x) != backgroundColor)
			{
				dst(y, x) = 0;
			}
		}
	}
	return dst;
}

Mat matrix_not(Mat_<uchar> src, uchar backgroundColor)
{
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src(y, x) == backgroundColor)
			{
				dst(y, x) = 0;
			}
		}
	}
	return dst;
}

bool matEquals(Mat_<uchar> m1, Mat_<uchar> m2)
{
	int height = m1.rows;
	int width = m1.cols;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (m1(y, x) != m2(y, x))
			{
				return false;
			}
		}
	}
	return true;
}

Mat generateStruc4()
{
	Mat_<uchar> struc = Mat(cv::Size(3, 3), CV_8UC1, Scalar(255));
	struc(0, 1) = 0;
	struc(1, 0) = 0;
	struc(1, 2) = 0;
	struc(2, 1) = 0;
	struc(1, 1) = 0;
	return struc;
}

void matrixDillation(int n)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> rst = img;

		for (int i = 1; i <= n; i++)
		{
			rst = matrix_dillate(rst, struc, 255);
			
		}
		imshow("orig", img);
		imshow("result", rst);
		waitKey();
	}
}

void matrixErosion(int n)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> rst = img;

		for (int i = 1; i <= n; i++)
		{
			rst = matrix_erode(rst, struc, 255);

		}
		imshow("orig", img);
		imshow("result", rst);
		waitKey();
	}
}

void imageOpening(int n)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> rst = img;
		Mat_<uchar> rst2 = img;

		for (int i = 1; i <= n; i++)
		{
			rst = matrix_erode(rst2, struc, 255);
			rst2 = matrix_dillate(rst, struc, 255);

		}
		imshow("orig", img);
		imshow("result", rst2);

		waitKey();
	}
}

void imageClosing(int n)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> rst = img;
		Mat_<uchar> rst2 = img;

		for (int i = 1; i < n; i++)
		{
			rst = matrix_dillate(rst2, struc, 255);
			rst2 = matrix_erode(rst, struc, 255);
		}
		imshow("orig", img);
		imshow("result", rst2);

		waitKey();
	}
}

void imageBoundry()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> rst = matrix_erode(img, struc, 255);
		Mat_<uchar> rst2 = matrix_difference(img, rst, 255);
		imshow("orig", img);
		imshow("result", rst2);

		waitKey();
	}
}

void imageFilling()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		int width = img.cols;
		int height = img.rows;
		Mat_<uchar> struc = generateStruc4();
		Mat_<uchar> complement = matrix_not(img, 255);
		Mat_<uchar> x0 = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
		Mat_<uchar> x1 = Mat(cv::Size(width, height), CV_8UC1, Scalar(255));
		x1(80, 80) = 0;

		while (!matEquals(x0, x1))
		{
			x1.copyTo(x0);
			Mat_<uchar> dil = matrix_dillate(x1, struc, 255);
			matrix_and(dil, complement, 255).copyTo(x1);

		}
		x1 = matrix_union(x1, img, 255);
		imshow("orig", img);
		imshow("result", x1);

		waitKey();
	}
}

float meanValue(Mat_<uchar> img)
{		
	int height = img.rows;
	int width = img.cols;
	int intensitySum=0;
	int M = height*width;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			intensitySum += img(i,j);
		}
	}
	float mean = (float)intensitySum / M;
	return mean;
}

float StandardDeviation(Mat_<uchar> img, float mean)
{
	int height = img.rows;
	int width = img.cols;
	int M = height*width;
	float dev = 0.0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dev += (img(i, j) - mean)*(img(i, j) - mean);
		}
	}
	dev = (float) sqrt(dev/M);
	return dev;
}

Point2i getMaxIntensity(std::vector<int> hist)
{
	int maxInt = -1;
	int minInt = -1;

	for (int i = 0; i < hist.size(); i++) 
	{
		if (hist[i] > 0) 
		{
			maxInt = i;
			if (minInt == -1)
			{
				minInt = i;
			}
		}
	}
	return Point2i(minInt,maxInt);
}

void Statistics()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		float mean= meanValue(img);
		float dev = StandardDeviation(img, mean);
		printf("Mean Intensity is %f\n", meanValue(img));
		printf("Standard deviation is %f\n", dev);
	}
}

std::vector<int> calc_histogram_vector(Mat src)
{

	int height = src.rows;
	int width = src.cols;

	std::vector<int> histogram;
	for (int i = 0; i < 256; i++)
		histogram.push_back(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int v = (int)src.at<uchar>(i, j);
			histogram[v]++;
		}
	}
	return histogram;
}

std::vector<float> calc_histogram_vector_fdp(Mat src)
{

	int height = src.rows;
	int width = src.cols;

	std::vector<float> histogram;
	for (int i = 0; i < 256; i++)
		histogram.push_back(0.0f);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int v = (int)src.at<uchar>(i, j);
			histogram[v]++;
		}
	}
	for (int i = 0; i < 256; i++)
		histogram[i] /= 1.0*(width*height);
	return histogram;
}

void adaptiveThreshold()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = src.clone();
		int height = src.rows;
		int width = src.cols;

		std::vector<int> histogram = calc_histogram_vector(src);
		Point2i maxMinPoint = getMaxIntensity(histogram);

		int max_intensity = maxMinPoint.y;
		int min_intensity = maxMinPoint.x;

		int T = (max_intensity + min_intensity) / 2;
		
		float error = 0.1; 
		int tk1 = T;
		int tk0 = T;
		do
		{
			tk0 = tk1;
			int ug1 = 0; 
			int n1 = 0;
			
			for (int i = min_intensity; i <= tk1; i++) 
			{
				n1 += histogram[i];
				ug1 += i*histogram[i];
			}
			ug1 /= n1;

			int ug2 = 0; 
			int n2 = 0;
			
			for (int i = tk1 + 1; i <= max_intensity; i++) 
			{
				n2 += histogram[i];
				ug2 += i * histogram[i];
			}
			ug2 /= n2;
			tk1 = (ug1 + ug2) / 2;
		} while (1.0*abs(tk1 - tk0) >= error);

		threshold(src, dst, tk1, 255, THRESH_BINARY);
		printf("%d\n",tk1);
		imshow("original", src);
		imshow("threshold", dst);
		waitKey();
	}
}

void shrinkImage()
{
	int gmin = 10;
	int gmax = 250;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = src.clone();
		int height = src.rows;
		int width = src.cols;

		std::vector<int> histogram = calc_histogram_vector(src);
		Point2i minMaxIntensity = getMaxIntensity(histogram);

		int maxIntensity = minMaxIntensity.y;
		int minIntensity = minMaxIntensity.x;

		float factor = 1.0*(gmax - gmin) / (maxIntensity - minIntensity);

		std::vector<int> histogram_out(256, 0);
		std::vector<int> mapper(256, 0);

		for (int i = 0; i<256; i++)
		{
			int indx = max(min(gmin + (i - minIntensity)*factor, 255), 0);
			histogram_out[indx] += histogram[i];
			mapper[i] = indx;
		}

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dst(y, x) = mapper[src(y, x)];
			}
		}
		
		int *sh_histogram_in = new int[256]();
		int *sh_histogram_out = new int[256]();
		
		for (int i = 0; i<256; i++)
		{
			sh_histogram_out[i] = histogram_out[i];
			sh_histogram_in[i] = histogram[i];
		
		}

		showHistogram("Histogram src", sh_histogram_in, 256, 256);
		showHistogram("Histogram dst", sh_histogram_out, 256, 256);

		imshow("src", src);
		imshow("dst", dst);
		waitKey();
		delete[] sh_histogram_in;
		delete[] sh_histogram_out;
	}
}

void gamma_correction()
{
	float omega = 1.3f;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = src.clone();
		int height = src.rows;
		int width = src.cols;
		std::vector<int> histogram = calc_histogram_vector(src);
		std::vector<int> histogram_out(256, 0);
		std::vector<int> mapper(256, 0);

		for (int i = 0; i < 256; i++)
		{
			int indx;
			indx = 255.0 * pow((i / 255.0), omega);
			histogram_out[indx] += histogram[i];
			mapper[i] = indx;
		}

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dst(y, x) = mapper[src(y, x)];
			}
		}
		int *sh_histogram_in = new int[256]();
		int *sh_histogram_out = new int[256]();

		for (int i = 0; i < 256; i++)
		{
			sh_histogram_out[i] = histogram_out[i];
			sh_histogram_in[i] = histogram[i];
		}
		showHistogram("Histogram IN", sh_histogram_in, 256, 256);
		showHistogram("Histogram OUT", sh_histogram_out, 256, 256);


		imshow("src", src);
		imshow("dst", dst);
		waitKey();
		delete[] sh_histogram_in;
		delete[] sh_histogram_out;
	}
}

void equalize_histogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = src.clone();
		int height = src.rows;
		int width = src.cols;
		std::vector<int> histogram = calc_histogram_vector(src);
		std::vector<float> histogram_fdp = calc_histogram_vector_fdp(src);

		std::vector<int> mapper(256, 0);

		for (int i = 1; i < 256; i++)
		{
			histogram_fdp[i] += histogram_fdp[i - 1];
		}

		for (int i = 0; i < 256; i++)
		{
			int indx = max(min(255 * histogram_fdp[i], 255), 0);
			mapper[i] = indx;
		}


		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dst(y, x) = mapper[src(y, x)];
			}
		}
		int *sh_histogram_in = new int[256]();
		for (int i = 0; i < 256; i++)
		{
			sh_histogram_in[i] = histogram[i];
		}
		showHistogram("Histogram IN", sh_histogram_in, 256, 256);


		imshow("src", src);
		imshow("dst", dst);
		waitKey();
		delete[] sh_histogram_in;
	}
}

int apply_filter_over_point(Mat_<uchar> src, Mat_<float> conv, Point2i point)
{
	int heightStruct = conv.rows;
	int widthStruct = conv.cols;
	int width = src.cols;
	int height = src.rows;

	Point2i strucCenter(widthStruct / 2, heightStruct / 2);
	float pointIntensity = 0;

	for (int y = 0; y < heightStruct; y++)
	{
		for (int x = 0; x < widthStruct; x++)
		{
			int dx = point.x - (strucCenter.x - x);
			int dy = point.y - (strucCenter.y - y);

			if (dx < 0 || dy < 0 || dx >= width || dy >= height) continue;

			pointIntensity += (float)src(dy, dx) * conv(y, x);
		}
	}

	return (int)pointIntensity;
}

Mat getGaussianConvulationMatrix()
{
	Mat_<float> struc = Mat(cv::Size(3, 3), CV_32FC1, Scalar(0.0f));
	struc(0, 0) = 1 / 16.0;
	struc(0, 1) = 2 / 16.0;
	struc(0, 2) = 1 / 16.0;
	struc(1, 0) = 2 / 16.0;
	struc(1, 1) = 4 / 16.0;
	struc(1, 2) = 2 / 16.0;
	struc(2, 0) = 1 / 16.0;
	struc(2, 1) = 2 / 16.0;
	struc(2, 2) = 1 / 16.0;
	return struc;
}

Mat getLaplaceConvulationMatrix()
{
	Mat_<float> struc = Mat(cv::Size(3, 3), CV_32FC1, Scalar(0.0f));
	struc(0, 0) = 0.0;
	struc(0, 1) = -1.0;
	struc(0, 2) = 0.0;
	struc(1, 0) = -1.0;
	struc(1, 1) = 4.0;
	struc(1, 2) = -1.0;
	struc(2, 0) = 0.0;
	struc(2, 1) = -1.0;
	struc(2, 2) = 0.0;
	return struc;
}

void apply_filter_lowPass()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		
		Mat convolutionMatrix = getGaussianConvulationMatrix();
		int paddingWidth = convolutionMatrix.cols / 2;
		int paddingHeight = convolutionMatrix.rows / 2;
		
		Mat_<uchar> dst = Mat::zeros(cv::Size(width - paddingWidth, height - paddingHeight), CV_8UC1);

		for (int y = paddingHeight; y < height - paddingHeight; y++)
		{
			for (int x = paddingWidth; x < width - paddingWidth; x++)
			{
				dst(y - paddingHeight, x - paddingWidth) =
					max(min(apply_filter_over_point(src, convolutionMatrix, Point2i(x, y)), 255), 0);
			}
		}
		imshow("Original", src);
		imshow("Filtered", dst);
		waitKey();
	}
}

void apply_filter_highPass()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		
		Mat convolutionMatrix = getLaplaceConvulationMatrix();
		int paddingWidth = convolutionMatrix.cols / 2;
		int paddingHeight = convolutionMatrix.rows / 2;
		
		Mat_<uchar> dst = Mat::zeros(cv::Size(width - paddingWidth, height - paddingHeight), CV_8UC1);

		for (int y = paddingHeight; y < height - paddingHeight; y++)
		{
			for (int x = paddingWidth; x < width - paddingWidth; x++)
			{
				dst(y - paddingHeight, x - paddingWidth) =
					abs(max(min(apply_filter_over_point(src, convolutionMatrix, Point2i(x, y)), 255), 0));
			}
		}
		imshow("Original", src);
		imshow("Filtered", dst);
		waitKey();
	}
}

void centering_transform(Mat_<float> img) 
{
	//expects floating point image
	for (int i = 0; i < img.rows; i++) 
	{
		for (int j = 0; j < img.cols; j++)
		{
			img(i, j) = ((i + j) & 1) ? -(img(i, j)) : img(i, j);
		}
	}
}

void apply_ideal_low_pass(Mat_<float> channels[2], float radius)
{
	int Hdiv2 = channels[0].rows / 2;
	int Wdiv2 = channels[0].cols / 2;
	for (int y = 0; y < channels[0].rows; y++)
	{ 
		for (int x = 0; x < channels[0].cols; x++)
		{
			if ((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x) > radius*radius)
			{
				channels[0](y, x) = 0;
			}
		}
	}

	Hdiv2 = channels[1].rows / 2;
	Wdiv2 = channels[1].cols / 2;
	for (int y = 0; y < channels[1].rows; y++)
	{
		for (int x = 0; x < channels[1].cols; x++)
		{
			if ((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x) > radius*radius)
			{
				channels[1](y, x) = 0;
			}
		}
	}
}

void apply_ideal_high_pass(Mat_<float> channels[2], float radius)
{
	int Hdiv2 = channels[0].rows / 2;
	int Wdiv2 = channels[0].cols / 2;
	for (int y = 0; y < channels[0].rows; y++)
	{
		for (int x = 0; x < channels[0].cols; x++)
		{
			if ((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x) <= radius*radius)
			{
				channels[0](y, x) = 0;
			}
		}
	}

	Hdiv2 = channels[1].rows / 2;
	Wdiv2 = channels[1].cols / 2;
	for (int y = 0; y < channels[1].rows; y++)
	{
		for (int x = 0; x < channels[1].cols; x++)
		{
			if ((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x) <= radius*radius)
			{
				channels[1](y, x) = 0;
			}
		}
	}
}

void apply_ideal_gaussian_low_pass(Mat_<float> channels[2], float amplitude)
{
	int Hdiv2 = channels[0].rows / 2;
	int Wdiv2 = channels[0].cols / 2;
	for (int y = 0; y < channels[0].rows; y++)
	{
		for (int x = 0; x < channels[0].cols; x++) 
		{
			float de = (-1.0)*((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x)) / (amplitude*amplitude);
			channels[0](y, x) = channels[0](y, x) *(exp(de));
		}
	}
	Hdiv2 = channels[1].rows / 2;
	Wdiv2 = channels[1].cols / 2;
	for (int y = 0; y < channels[1].rows; y++)
	{ 
		for (int x = 0; x < channels[1].cols; x++) 
		{
			float de = (-1.0)*((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x)) / (amplitude*amplitude);
			channels[1](y, x) = channels[1](y, x) *(exp(de));
		}
	}
}

void apply_ideal_gaussian_high_pass(Mat_<float> channels[2], float amplitude)
{
	int Hdiv2 = channels[0].rows / 2;
	int Wdiv2 = channels[0].cols / 2;
	for (int y = 0; y < channels[0].rows; y++)
	{
		for (int x = 0; x < channels[0].cols; x++)
		{
			float de = (-1.0)*((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x)) / (amplitude*amplitude);
			channels[0](y, x) = channels[0](y, x) *(1 - exp(de));
		}
	}
	Hdiv2 = channels[1].rows / 2;
	Wdiv2 = channels[1].cols / 2;
	for (int y = 0; y < channels[1].rows; y++)
	{
		for (int x = 0; x < channels[1].cols; x++)
		{
			float de = (-1.0)*((Hdiv2 - y)*(Hdiv2 - y) + (Wdiv2 - x)*(Wdiv2 - x)) / (amplitude*amplitude);
			channels[1](y, x) = channels[1](y, x) *(1 - exp(de));
		}
	}
}

enum FilterType { LOW_PASS, HIGH_PASS, GAUSSIAN_HIGH_PASS, GAUSSIAN_LOW_PASS };

void generic_frequency_domain_filter(FilterType type) 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat srcf;
		src.convertTo(srcf, CV_32FC1);
		//centering transformation
		centering_transform(srcf);
		//perform forward transform with complex image output
		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
		//split into real and imaginary channels
		Mat_<float> channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);  // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
								   //calculate magnitude and phase in floating point images mag and phi
		Mat_<float> mag, phi;

		magnitude(channels[0], channels[1], mag);
		phase(channels[0], channels[1], phi);

		for (int y = 0; y < mag.rows; y++)
		{
			for (int x = 0; x < mag.rows; x++)
			{
				mag(y, x) = log(mag(y, x) + 1);
			}
		}

		Mat_<float> normalized_mag, normalized_phi;
		normalize(mag, normalized_mag, 0, 1, NORM_MINMAX, CV_32F);
		normalize(phi, normalized_phi, 0, 1, NORM_MINMAX, CV_32F);

		float radius = 10.0f; float amplitude = 10.0f;
		//display the phase and magnitude images here
		imshow("Magnitude Image", normalized_mag);
		imshow("Phase Image", normalized_phi);
		//insert filtering operations on Fourier coefficients here
		switch (type)
		{
		case LOW_PASS:
			apply_ideal_low_pass(channels, radius);
			break;
		case HIGH_PASS:
			apply_ideal_high_pass(channels, radius);
			break;
		case GAUSSIAN_LOW_PASS:
			apply_ideal_gaussian_low_pass(channels, amplitude);
			break;
		case GAUSSIAN_HIGH_PASS:
			apply_ideal_gaussian_high_pass(channels, amplitude);
			break;
		}

		//perform inverse transform and put results in dstf
		Mat dst, dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		//inverse centering transformation
		centering_transform(dstf);
		//normalize the result and put in the destination image
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("dst", dst);
		waitKey();
	}
}

std::vector<uchar> get_convolution_neighbours(Mat_<uchar> src, Mat_<uchar> kernel, Point2i localPoint)
{
	std::vector<uchar> neighbours;
	int heightStruct = kernel.rows;
	int widthStruct = kernel.cols;
	int height = src.rows;
	int width = src.cols;

	Point2i strucCenter(widthStruct / 2, heightStruct / 2);

	for (int y = 0; y < heightStruct; y++)
		for (int x = 0; x < widthStruct; x++) 
		{
			int dx = localPoint.x - (strucCenter.x - x);
			int dy = localPoint.y - (strucCenter.y - y);

			if (dx < 0 || dy < 0 || dx >= width || dy >= height) continue;

			neighbours.push_back(src(dy, dx));
		}
	return neighbours;
}

enum NoiseFilterType { MEDIAN, GAUSSIAN };

Mat filter_median(Mat_<uchar> src, Mat_<uchar> kernel)
{
	Mat_<uchar> dst = src.clone();
	int height = src.rows;
	int width = src.cols;
	int pd_height = kernel.rows / 2;
	int pd_width = kernel.cols / 2;
	
	for (int y = pd_height; y < height - pd_height; y++)
		for (int x = pd_width; x < width - pd_width; x++)
		{
			std::vector<uchar>neighbours = get_convolution_neighbours(src, kernel, Point2i(x, y));
			std::sort(neighbours.begin(), neighbours.end());
			dst(y, x) = neighbours[neighbours.size() / 2];
		}
	return dst;
}

Mat filter_gaussian(Mat_<uchar> src, int kernel_size)
{
	Mat_<float> filter = Mat(cv::Size(kernel_size, kernel_size), CV_32FC1, Scalar(0));
	int height = src.rows;
	int width = src.cols;
	int pd = kernel_size / 2;
	float std_dev = kernel_size / 6.0;
	float cst = 2 * std_dev * std_dev;

	for (int y = 0; y < kernel_size; y++)
		for (int x = 0; x < kernel_size; x++) 
		{
			filter(y, x) = (1.0 / (pi*cst))*exp(-((x - pd)*(x - pd) + (y - pd)*(y - pd)) / cst);
		}

	Mat_<uchar> dst = src.clone();
	for (int y = pd; y < height - pd; y++)
		for (int x = pd; x < width - pd; x++)
		{
			dst(y, x) = max(min(apply_filter_over_point(src, filter, Point2i(x, y)), 255), 0);
		}
	return dst;
}

Mat filter_gaussian_1d(Mat_<uchar> src, int kernel_size)
{
	Mat_<float> filter_x = Mat(cv::Size(kernel_size, 1), CV_32FC1, Scalar(0));
	Mat_<float> filter_y = Mat(cv::Size(1, kernel_size), CV_32FC1, Scalar(0));
	int height = src.rows;
	int width = src.cols;
	int pd = kernel_size / 2;
	float std_dev = kernel_size / 6.0;
	float cst = 2 * std_dev * std_dev;

	for (int x = 0; x < filter_x.cols; x++) 
	{
		filter_x(0, x) = (1.0 / (sqrt(pi* 2.0)* std_dev))*exp(-((x - pd)*(x - pd)) / cst);
	}

	for (int y = 0; y < filter_y.rows; y++) 
	{
		filter_y(y, 0) = (1.0 / (sqrt(pi* 2.0)* std_dev))*exp(-((y - pd)*(y - pd)) / cst);
	}

	Mat_<uchar> dst = src.clone();
	for (int y = pd; y < height - pd; y++)
		for (int x = pd; x < width - pd; x++)
		{
			dst(y, x) = max(min(apply_filter_over_point(src, filter_x, Point2i(x, y)), 255), 0);

		}

	Mat_<uchar> dst2 = dst.clone();
	for (int y = pd; y < height - pd; y++)
		for (int x = pd; x < width - pd; x++)
		{
			dst2(y, x) = max(min(apply_filter_over_point(dst, filter_y, Point2i(x, y)), 255), 0);
		}
	return dst2;
}

void filter_noise(int kernel_size = 5)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> kernel = Mat(cv::Size(kernel_size, kernel_size), CV_8UC1, Scalar(0));

		Mat_<uchar> dst = filter_median(src, kernel);
		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void filter_noise_gaussian(int kernel_size = 7)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> dst = filter_gaussian(src, kernel_size);
		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void filter_noise_gaussian_1d(int kernel_size = 7)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> dst = filter_gaussian_1d(src, kernel_size);
		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void compare_execution()
{

	const int kernel_size = 7;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		double t = (double)getTickCount();
		filter_gaussian(src, kernel_size);
		t = ((double)getTickCount() - t) / getTickFrequency();
		double time_gaussian_2d = t * 1000;

		t = (double)getTickCount();
		filter_gaussian_1d(src, kernel_size);
		t = ((double)getTickCount() - t) / getTickFrequency();
		double time_gaussian_1d = t * 1000;

		printf("Time Gaussian 2D = %.3f [ms]\n", time_gaussian_2d);
		printf("Time Gaussian 1D = %.3f [ms]\n", time_gaussian_1d);
		getchar();
		getchar();
	}
}
int main()
{
	int factor;
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Increase grey level by given factor\n");
		printf(" 11 - Multiply Grey level by a given factor\n");
		printf(" 12 - Color image\n");
		printf(" 13 - Float matrix\n");
		printf(" 14 - Make 3 channels\n");
		printf(" 15 - Convert to greyscale\n");
		printf(" 16 - Convert to binary\n");
		printf(" 17 - convert to HSV\n");
		printf(" 18 - in inside\n");
		printf(" 19 - make Histogram \n");
		printf(" 20 - make Histogram with m bins\n");
		printf(" 21 - Floyd Steinberg\n");
		printf(" 22 - Process Objects\n");
		printf(" 23 - Breath First Search Labeling\n");
		printf(" 24 - Equivalance Class Labelling\n");
		printf(" 25 - Border Tracing\n");
		printf(" 26 - Reconstruct from txt\n");
		printf(" 27 - Dilate\n");
		printf(" 28 - Erode\n");
		printf(" 29 - Opening...\n");
		printf(" 30 - Closing...\n");
		printf(" 31 - Boundary Extraction\n");
		printf(" 32 - Region Filling\n");
		printf(" 33 - compute Mean Value\n");
		printf(" 34 - Adaptive Thresholding\n");
		printf(" 35 - Shrinking Image\n");
		printf(" 36 - Gamma correction\n");
		printf(" 37 - Equalize histograms\n");
		printf(" 38 - Apply Low Pass Filter\n");
		printf(" 39 - Apply High Pass Filter\n");
		printf(" 40 - Generic Frequency Domain Filter\n");
		printf(" 41 - Median Filter\n");
		printf(" 42 - Compare filters\n");
		printf(" 43 - Filter Noise Gaussian\n");
		printf(" 44	 - Filter Noise Gaussian 1d\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
			{
				scanf("%d", &factor);
				IncreaseGrayLevel(factor);
			}
			case 11:
			{
				scanf("%d", &factor);
				MultiplicativeGray(factor);
			}
			case 12:
				CreateColorImag();
			case 13:
				MakeFloatMatrix();
			case 14:
				Make3Channels();
			case 15:
				ToGrayScale();
			case 16:
				ToBinary();
			case 17:
				toHSV();
			case 18:
				isInside();
			case 19:
				makeHisto();
			case 20:
			{
				scanf("%d", &factor);
				makeHistoWithBins(factor);
			}
			case 21:
				floyd_steinberg_dithering();
			case 22:
				process_objects();
			case 23:
				breadth_first_labeling();
			case 24:
				equivalence_class_labeling();
			case 25:
				countours(EIGHT);
			case 26:
				reconstruct_from_text();
			case 27:
				scanf("%d", &factor);
				matrixDillation(factor);
			case 28:
				scanf("%d", &factor);
				matrixErosion(factor);
			case 29:
				scanf("%d", &factor);
				imageOpening(factor);
			case 30: 
				scanf("%d", &factor);
				imageClosing(factor);
			case 31:
				imageBoundry();
			case 32:
				imageFilling();
			case 33:
				Statistics();
			case 34:
				adaptiveThreshold();
			case 35:
				shrinkImage();
			case 36:
				gamma_correction();
			case 37:
				equalize_histogram();
			case 38:
				apply_filter_lowPass();
			case 39:
				apply_filter_highPass();
			case 40:
				generic_frequency_domain_filter(LOW_PASS); //  LOW_PASS, HIGH_PASS, GAUSSIAN_HIGH_PASS, GAUSSIAN_LOW_PASS
			case 41:
				filter_noise();
			case 42:
				compare_execution();
			case 43:
				filter_noise_gaussian();
				break;
			case 44:
				filter_noise_gaussian_1d();
		}
	}
	while (op!=0);
	return 0;
}