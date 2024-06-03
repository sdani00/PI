// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <random>

#include "common.h"

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
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
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
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
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

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
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
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
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
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
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

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

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
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
	//	setMouseCallback("My Window", MyCallBackFunc, &src);

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
	for (int i = 0; i < hist_cols; i++)
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


void additiveFactor(int factor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
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

				int sum = val + factor;

				if (sum > 255)
					val = 255;
				else if (sum < 0)
					val = 0;
				else
					val = sum;

				dst.at<uchar>(i, j) = val;
			}
		}

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void multiplyFactor(int i)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
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

				int sum = val * i;

				if (sum > 255)
					val = 255;
				else if (sum < 0)
					val = 0;
				else
					val = sum;

				dst.at<uchar>(i, j) = val;
			}
		}

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void createImage()
{
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			if (i < 128 && j < 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (i < 128 && j >= 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
			else if (i >= 128 && j < 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
			else
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
			}
		}
	}
	imshow("image", img);
	waitKey();
}

void inverseMatrix()
{
	Mat A = (Mat_<float>(3, 3) << 1, 2, 3,
		0, 1, 4,
		5, 6, 0);
	Mat invA;
	int determinant = A.at<float>(0, 0) * (A.at<float>(1, 1) * A.at<float>(2, 2) - A.at<float>(1, 2) * A.at<float>(2, 1)) -
		A.at<float>(0, 1) * (A.at<float>(1, 0) * A.at<float>(2, 2) - A.at<float>(1, 2) * A.at<float>(2, 0)) +
		A.at<float>(0, 2) * (A.at<float>(1, 0) * A.at<float>(2, 1) - A.at<float>(1, 1) * A.at<float>(2, 0));

	Mat transpose = (Mat_<float>(3, 3) << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
		A.at<float>(0, 1), A.at<float>(1, 1), A.at<float>(2, 1),
		A.at<float>(0, 2), A.at<float>(1, 2), A.at<float>(2, 2));

	Mat complementary = (Mat_<float>(3, 3) << A.at<float>(1, 1) * A.at<float>(2, 2) - A.at<float>(1, 2) * A.at<float>(2, 1),
		A.at<float>(1, 0) * A.at<float>(2, 2) - A.at<float>(1, 2) * A.at<float>(2, 0),
		A.at<float>(1, 0) * A.at<float>(2, 1) - A.at<float>(1, 1) * A.at<float>(2, 0),
		A.at<float>(0, 1) * A.at<float>(2, 2) - A.at<float>(0, 2) * A.at<float>(2, 1),
		A.at<float>(0, 0) * A.at<float>(2, 2) - A.at<float>(0, 2) * A.at<float>(2, 0),
		A.at<float>(0, 0) * A.at<float>(2, 1) - A.at<float>(0, 1) * A.at<float>(2, 0),
		A.at<float>(0, 1) * A.at<float>(1, 2) - A.at<float>(0, 2) * A.at<float>(1, 1),
		A.at<float>(0, 0) * A.at<float>(1, 2) - A.at<float>(0, 2) * A.at<float>(1, 0),
		A.at<float>(0, 0) * A.at<float>(1, 1) - A.at<float>(0, 1) * A.at<float>(1, 0));

	invA = (1 / determinant) * complementary;

	printf("Inverse matrix: \n");
	printf("%f %f %f\n", invA.at<float>(0, 0), invA.at<float>(0, 1), invA.at<float>(0, 2));
	printf("%f %f %f\n", invA.at<float>(1, 0), invA.at<float>(1, 1), invA.at<float>(1, 2));
	printf("%f %f %f\n", invA.at<float>(2, 0), invA.at<float>(2, 1), invA.at<float>(2, 2));
}

void splitImage()
{
	char fname[MAX_PATH];

	openFileDlg(fname);

	Mat img = imread(fname, IMREAD_COLOR);

	imshow("COLOR", img);

	Mat R(img.rows, img.cols, CV_8UC3);
	Mat G(img.rows, img.cols, CV_8UC3);
	Mat B(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			R.at<Vec3b>(i, j) = Vec3b(0, 0, pixel[0]);
			G.at<Vec3b>(i, j) = Vec3b(0, pixel[1], 0);
			B.at<Vec3b>(i, j) = Vec3b(pixel[2], 0, 0);
		}
	}

	imshow("RED", R);
	imshow("GREEN", G);
	imshow("BLUE", B);

	waitKey(0);
}

void fromColorToGrayscale()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, IMREAD_COLOR);
	imshow("COLOR", img);

	Mat gray(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			gray.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}

	imshow("GRAYSCALE", gray);
	waitKey(0);
}

void fromGrescaleToBinary(int threshold)
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("GRAYSCALE", img);

	Mat binary(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			uchar grey_level = img.at<uchar>(i, j);

			if (grey_level < threshold)
				binary.at<uchar>(i, j) = 0;
			else
				binary.at<uchar>(i, j) = 255;
		}
	}

	imshow("BINARY", binary);
	waitKey(0);
}

void fromRGBtoHSV()
{
	float b, g, r;
	float M, m;
	float H, S, V, C;

	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, IMREAD_COLOR);
	imshow("COLOR", img);

	Mat H_mat(img.rows, img.cols, CV_8UC1);
	Mat S_mat(img.rows, img.cols, CV_8UC1);
	Mat V_mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			b = (float)pixel[0] / 255;
			g = (float)pixel[1] / 255;
			r = (float)pixel[2] / 255;

			M = max(max(b, g), max(b, r));
			m = min(min(b, g), min(b, r));

			C = M - m;
			V = M;

			if (V != 0)
			{
				S = C / V;
			}
			else
			{
				S = 0;
			}

			if (C != 0)
			{
				if (M == r)
				{
					H = 60 * (g - b) / C;
				}
				else if (M == g)
				{
					H = 120 + 60 * (b - r) / C;
				}
				else if (M == b)
				{
					H = 240 + 60 * (r - g) / C;
				}
			}
			else
			{
				H = 0;
			}

			if (H < 0)
			{
				H = H + 360;
			}

			uchar H_norm = (H * 255) / 360;
			uchar S_norm = S * 255;
			uchar V_norm = V * 255;

			H_mat.at<uchar>(i, j) = H_norm;
			S_mat.at<uchar>(i, j) = S_norm;
			V_mat.at<uchar>(i, j) = V_norm;
		}

	imshow("HUE", H_mat);
	imshow("SATURATION", S_mat);
	imshow("VALUE", V_mat);
	waitKey(0);
}

void rgb_to_hsv() {
	float b, g, r, M, m, H, S, V, C;
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_COLOR);
	imshow("COLOR", img);

	Mat H_mat(img.rows, img.cols, CV_8UC1);
	Mat S_mat(img.rows, img.cols, CV_8UC1);
	Mat V_mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at< Vec3b>(i, j);
			b = (float)pixel[0] / 255;
			g = (float)pixel[1] / 255;
			r = (float)pixel[2] / 255;

			M = max(max(b, g), max(b, r));
			m = min(min(b, g), min(b, r));

			C = M - m;
			V = M;

			if (V != 0)
				S = C / V;
			else // negru
				S = 0;

			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else // grayscale
				H = 0;
			if (H < 0)
				H = H + 360;

			uchar H_norm = H * 255 / 360;
			uchar S_norm = S * 255;
			uchar V_norm = V * 255;
			H_mat.at<unsigned char>(i, j) = H_norm;
			S_mat.at<unsigned char>(i, j) = S_norm;
			V_mat.at<unsigned char>(i, j) = V_norm;
		}
	}

	imshow("H", H_mat);
	imshow("S", S_mat);
	imshow("V", V_mat);
	waitKey(0);
}

int isInside(Mat img, int i, int j) {

	if (i >= img.rows || i < 0) {
		return 0;
	}

	if (j >= img.cols || j < 0) {
		return 0;
	}

	return 1;
}

int* histo(Mat img)
{
	int* h = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			h[img.at<uchar>(i, j)]++;
		}
	}

	return h;
}

float* FDP(Mat img) {
	int* h = histo(img);

	float* fdp = (float*)calloc(256, sizeof(float));
	float M = img.rows * img.cols;

	for (int i = 0; i < 256; i++)
	{
		fdp[i] = h[i] / M;
	}

	return fdp;
}

int* reduct_histo(Mat img, int acc)
{
	float k = 255 / acc;
	int* h = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int var = img.at<uchar>(i, j) / k;
			h[var]++;
		}
	}

	return h;
}

Mat readImage(char fname[MAX_PATH], int greyscale)
{
	openFileDlg(fname);
	Mat img;

	if (greyscale)
	{
		img = imread(fname, IMREAD_GRAYSCALE);
	}
	else
	{
		img = imread(fname, IMREAD_COLOR);
	}

	return img;
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}


bool neighbours(Mat* img, int i, int j) {
	int dx[] = { -1, 1, -1, 0, 0, 1, 1, 1 };
	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	for (int d = 0; d < 8; d++) {
		if (img->at<Vec3b>(i, j) != img->at<Vec3b>(i + dx[d], j + dy[d]))
		{
			return true;
		}
	}
	return false;
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	Vec3b pixel;
	int arie = 0;
	int centerR = 0;
	int centerC = 0;
	float axaUp = 0;
	float axaDown = 0;
	float phi = 0;
	int perimetru = 0;
	float elongatia = 0;
	int imin = src->rows, jmin = src->cols, imax = 0, jmax = 0;

	Mat copy = src->clone();
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		pixel = src->at<Vec3b>(y, x);

		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++) {
				if (pixel == src->at<Vec3b>(i, j)) {
					arie++;
					centerR += i;
					centerC += j;

					if (i < imin)
						imin = i;
					if (i > imax)
						imax = i;
					if (j < jmin)
						jmin = j;
					if (j > jmax)
						jmax = j;
				}
			}
		}

		centerR /= arie;
		centerC /= arie;
		printf("Aria: %d\n", arie);
		printf("Coordonatele centrului: (%d, %d)\n", centerR, centerC);
		copy.at<Vec3b>(centerR, centerC) = Vec3b(0, 0, 0);

		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++) {
				if (pixel == src->at<Vec3b>(i, j)) {
					axaUp = (i - centerR) * (j - centerC);
					axaDown = pow((j - centerC), 2) - pow((i - centerR), 2);

					if (neighbours(src, i, j)) {
						perimetru++;
						copy.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					}
				}
			}
		}
		perimetru *= (CV_PI / 4);
		axaUp = axaUp * 2;
		float axaResult = atan2(axaUp, axaDown);
		phi = axaResult / 2 * (180 / CV_PI);
		line(copy, Point(imax, jmin), Point(imin, jmax), Scalar(2), 1);

		printf("Axa de alungire: %f\n", phi);
		printf("Perimetru: %d\n", perimetru);
		float factor_de_subtiere = 4 * CV_PI * arie / pow(perimetru, 2);
		printf("Factor de subtiere: %f\n", factor_de_subtiere);
		elongatia = (float)(jmax - jmin + 1) / (imax - imin + 1);
		printf("Elongatia: %f\n", elongatia);
		imshow("copy", copy);
	}
}

// LAB 5

void afiseza_imagine_etichetata(Mat labels, int label_count) {
	std::uniform_int_distribution<int> d(0, 255);
	std::default_random_engine gen;

	Vec3b* colors = (Vec3b*)calloc(label_count + 1, sizeof(Vec3b));
	colors[0] = Vec3b(255, 255, 255);

	for (int i = 1; i <= label_count; i++)
	{
		int R = d(gen);
		int G = d(gen);
		int B = d(gen);
		colors[i] = Vec3b(B, G, R);
	}


	Mat result(labels.rows, labels.cols, CV_8UC3);
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<Vec3b>(i, j) = colors[labels.at<uchar>(i, j)];
		}
	}

	imshow("result", result);
	waitKey();
}

void traversare_in_latime()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("img", img);

	// etichete, initial sunt 0 pentru ca nu am etichetat nimic
	int label_count = 0;
	Mat labels = Mat::zeros(img.rows, img.cols, CV_8UC1);

	int type_neighbor = 8;
	int di[] = { -1, 1, -1, 0, 0, 1, -1, 1 };
	int dj[] = { -1, 0,  1, -1, 1, -1, 0, 1 };

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{ // am gasit un varf neetichetat
				label_count++;
				std::queue<Point> q;
				labels.at<uchar>(i, j) = label_count;
				q.push({ i, j });

				while (!q.empty())
				{
					Point p = q.front();
					q.pop();
					for (int k = 0; k < type_neighbor; k++)
					{
						int i_neighbor = p.x + di[k];
						int j_neighbor = p.y + dj[k];

						if (isInside(labels, i_neighbor, j_neighbor))
						{
							if (img.at<uchar>(i_neighbor, j_neighbor) == 0 && labels.at<uchar>(i_neighbor, j_neighbor) == 0)
							{
								labels.at<uchar>(i_neighbor, j_neighbor) = label_count;
								q.push({ i_neighbor, j_neighbor });
							}
						}
					}
				}
			}
		}
	}

	afiseza_imagine_etichetata(labels, label_count);
}


void etichetare_clase_de_echivalenta()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("img", img);

	// etichete, initial sunt 0 pentru ca nu am etichetat nimic
	int label_count = 0;
	Mat labels = Mat::zeros(img.rows, img.cols, CV_8UC1);


	int di[] = { -1, -1, -1,  0 };
	int dj[] = { -1,  0,  1, -1 };

	std::vector<std::vector<int>> edges;
	edges.resize(256); // initial maxim atatea etichete putem sa retinem

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{ // am gasit un varf neetichetat
				std::vector<int> L; // lista de etichete din vecinatate
				for (int k = 0; k < 4; k++)
				{
					int i_neighbor = i + di[k];
					int j_neighbor = j + dj[k];
					if (labels.at<uchar>(i_neighbor, j_neighbor) != 0) { // sunt etichete in vecinatate
						L.push_back(labels.at<uchar>(i_neighbor, j_neighbor));
					}
				}
				if (L.empty())
				{
					label_count++;
					labels.at<uchar>(i, j) = label_count;
				}
				else
				{ // sunt etichete trebuie sa o luam pe cea minima
					int min = L[0];
					for (int e = 0; e < L.size(); e++)
					{
						if (L[e] < min) {
							min = L[e];
						}
					}
					labels.at<char>(i, j) = min;
					for (int e = 0; e < L.size(); e++)
					{
						if (L[e] != min)
						{ //clasele sunt echivalente
							edges[min].push_back(L[e]);
							edges[L[e]].push_back(min);
						}
					}
				}
			}
		}
	}

	edges.resize(label_count + 1);

	int new_label_count = 0;
	int* new_labels = (int*)calloc(label_count + 1, sizeof(int));

	for (int i = 1; i <= label_count; i++)
	{
		if (new_labels[i] == 0)
		{
			new_label_count++;
			std::queue<int> Q;
			new_labels[i] = new_label_count;
			Q.push(i);
			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for each (int y in edges[x])
				{
					if (new_labels[y] == 0) {
						new_labels[y] = new_label_count;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < labels.rows; i++)
	{
		for (int j = 0; j < labels.cols; j++)
		{
			labels.at<uchar>(i, j) = new_labels[labels.at<uchar>(i, j)];
		}
	}

	afiseza_imagine_etichetata(labels, new_label_count);
}
boolean are_points_equal(Point p1, Point p2)
{
	if (p1.x == p2.x && p1.y == p2.y) {
		return true;
	}

	return false;
}

int new_dir(int old_dir)
{
	if (old_dir % 2 == 0)
	{
		return (old_dir + 7) % 8;
	}

	return (old_dir + 6) % 8;
}

void binary(Mat img, int threshold)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) < threshold)
			{
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void contur()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("img", img);
	binary(img, 128);
	int dir = 7;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };


	//Point p1, p2, p3, p4; // p1 primul pixel obiect; p2 al doilea pixel
						  // p3 pixel curent; p4 urmatorul pixel curent
	bool found = false;

	Point start;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) { //primul pixel obiect
				start = Point(i, j);
				//p1 = Point(i, j);
				//p3 = p1;
				found = true;
			}
		}

		if (found)
			break;
	}

	std::vector<int> codes;
	std::vector<Point> contur;

	Point p = start;
	contur.push_back(p);

	bool give_up = false;
	while (true) {
		dir = new_dir(dir);
		int try_count = 0;

		while (img.at<uchar>(p.x + di[dir], p.y + dj[dir]) != 0)
		{ //cat timp nu gasim un pixel obiect vecin
			dir++;
			try_count++;

			if (dir == 8)
			{ // pp. ca in cele din urma se gaseste totusi un pixel obiect (adica nu avem pixel singulari in imagine)
				dir = 0;
			}

			if (try_count == 9)
			{ // am incercat deja toti vecinii -> ciclu infinit
				give_up = true;
				break;
			}
		}

		if (give_up)
		{
			break;
		}

		p.x = p.x + di[dir];
		p.y = p.y + dj[dir];

		codes.push_back(dir);
		contur.push_back(p);

		int size = contur.size();

		if (size > 2 && are_points_equal(contur[0], contur[size - 2])
			&& are_points_equal(contur[1], contur[size - 1]))
		{
			break;
		}
	}

	codes.pop_back();

	Mat result = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<uchar>(i, j) = 255;
		}
	}

	for (int i = 1; i < codes.size(); i++)
	{
		int d = codes[i];
		start.x = start.x + di[d];
		start.y = start.y + dj[d];

		result.at<uchar>(start.x, start.y) = 0;
	}

	std::cout << "Cod: ";
	for (int code : codes)
	{
		std::cout << code << " ";
	}

	std::cout << "." << std::endl;

	std::cout << "Derivata: ";
	std::vector<int> derivata;

	for (int i = 0; i < codes.size(); i++)
	{
		int deriv = codes[i] - codes[i + 1];

		if (deriv < 0)
		{
			deriv = deriv + 8;
		}

		derivata.push_back(deriv);
		std::cout << derivata[i] << " ";
	}
	std::cout << "." << std::endl;

	imshow("contur", result);
	waitKey();
}

void reconstruct_contur_from_file()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	std::vector<int> codes;
	std::ifstream file(fname);

	int start_i;
	file >> start_i;

	int start_j;
	file >> start_j;

	Point start = Point(start_i, start_j);

	int size;
	file >> size;

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };

	while (!file.eof())
	{
		int code;
		file >> code;
		codes.push_back(code);
	}

	Mat result = Mat(1024, 1024, CV_8UC1);

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<uchar>(i, j) = 255;
		}
	}

	for (int i = 1; i < codes.size(); i++)
	{
		int d = codes[i];
		start.x = start.x + di[d];
		start.y = start.y + dj[d];

		if (isInside(result, start.x, start.y))
		{
			result.at<uchar>(start.x, start.y) = 0;
		}
	}

	imshow("contur", result);
	waitKey();
}


// LAB 7

void dilatare() {
	uchar val;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);

	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) { // pixelii din kernel devin pixeli obiect
					dst.at<uchar>(i + di[k], j + dj[k]) = 0;
				}
			}
		}
	}
	imshow("dilatare", dst);
	waitKey(0);
}

void eroziune() {
	uchar val;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);

	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) == 255) {// gasim in kernel pixeli fundal
						dst.at<uchar>(i, j) = 255; // atunci pixelul obiect devine fundal
						break;
					}
				}
			}
		}
	}
	imshow("eroziune", dst);
	waitKey(0);
}



void deschidere() { // eroziune urmata de dilatare
	uchar val;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);

	// imaginea erodata
	Mat dst1 = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) == 255) {
						dst1.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	// imaginea dilatat
	Mat dst2 = dst1.clone();

	for (int i = 1; i < dst1.rows - 1; i++) {
		for (int j = 1; j < dst1.cols - 1; j++) {
			if (dst1.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 4; k++) {
					dst2.at<uchar>(i + di[k], j + dj[k]) = 0;
				}
			}
		}
	}

	imshow("deschidere", dst2);
	waitKey(0);
}

void inchidere() { // dilatare urmata de o eroziune 
	uchar val;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1, -1, -1, 0, 1 };
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);

	// imaginea dilatate
	Mat dst1 = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				if (src.at<uchar>(i, j) == 0) {
					for (int k = 0; k < 8; k++) {
						dst1.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}
	}

	// imaginea erodata
	Mat dst2 = dst1.clone();

	for (int i = 1; i < dst1.rows - 1; i++) {
		for (int j = 1; j < dst1.cols - 1; j++) {
			if (dst1.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					if (dst1.at<uchar>(i + di[k], j + dj[k]) == 255) {
						dst2.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	imshow("inchidere", dst2);
	waitKey(0);
}

//LAB 08

void info_histo()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	float media = 0;
	float deviatia_standard = 0;

	int histo[256] = {};
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histo[img.at<uchar>(i, j)]++;
			media += img.at<uchar>(i, j);
		}
	}

	float M = img.rows * img.cols;
	media = media / M;

	int histo_cumulativa[256] = {};
	histo_cumulativa[0] = histo[0];
	for (int i = 0; i < 256; i++) {
		if (i != 0) {
			histo_cumulativa[i] = histo_cumulativa[i - 1] + histo[i];
		}
		deviatia_standard += (pow(i - media, 2) * histo[i] / M);
	}

	deviatia_standard = pow(deviatia_standard, 1.0 / 2);

	std::cout << "Media: " << media << std::endl;
	std::cout << "Deviatia standard: " << deviatia_standard << std::endl;

	showHistogram("histograma", histo, 256, 256);
	showHistogram("histograma cumulativa", histo_cumulativa, 256, 256);
	waitKey();
}


void binarizare_automata() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int histo[256] = {};
	int imax = 0;
	int imin = 255;


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histo[img.at<uchar>(i, j)]++;

			if (img.at<uchar>(i, j) < imin) {
				imin = img.at<uchar>(i, j);
			}
			if (img.at<uchar>(i, j) > imax) {
				imax = img.at<uchar>(i, j);
			}
		}
	}


	int T = (imin + imax) / 2;
	int newT;
	while (true) {
		float media1 = 0;
		float media2 = 0;
		int N1 = 0;
		int N2 = 0;
		for (int i = 0; i < 256; i++) {
			if (i <= T) {
				media1 += i * histo[i];
				N1 += histo[i];
			}
			else {
				media2 += i * histo[i];
				N2 += histo[i];
			}
		}

		media1 = media1 / N1;
		media2 = media2 / N2;

		newT = (media1 + media2) / 2;

		if (newT - T < 0.1) {
			break;
		}
		T = newT;
	}

	std::cout << "T = " << T << std::endl;

	Mat binary = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < T) {
				binary.at<uchar>(i, j) = 0;
			}
			else {
				binary.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("binary", binary);
	waitKey();
}

void transformari_histograma() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int h[256] = {};
	int new_contrast[256] = {};

	int imin = 255;
	int imax = 0;

	int g_out_min = 10;
	int g_out_max = 250;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			h[img.at<uchar>(i, j)]++;

			if (img.at<uchar>(i, j) < imin) {
				imin = img.at<uchar>(i, j);
			}
			if (img.at<uchar>(i, j) > imax) {
				imax = img.at<uchar>(i, j);
			}
		}
	}

	float g = (float)(g_out_max - g_out_min) / (imax - imin);

	for (int i = 0; i < 256; i++) {
		new_contrast[i] = g_out_min + (i - imin) * g;
	}

	Mat contrast = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			contrast.at<uchar>(i, j) = new_contrast[img.at<uchar>(i, j)];
		}
	}

	float corectie_gama = 2;
	int gama[256] = {};
	for (int i = 0; i < 256; i++) {
		int val = 255 * pow(i / 255.0, corectie_gama);
		if (val < 0) {
			gama[i] = 0;
		}
		else if (val > 255) {
			gama[i] = 255;
		}
		else {
			gama[i] = val;
		}

	}

	Mat gamaM = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			gamaM.at<uchar>(i, j) = gama[img.at<uchar>(i, j)];
		}
	}
	//imshow("contrast", contrast);
	imshow("gama", gamaM);
	showHistogram("histograma", h, 256, 256);
	int* new_histo = histo(gamaM);
	showHistogram("new histogram", new_histo, 256, 256);
	waitKey();
}


void egalizarea_histogramei() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int* h = histo(img);
	float histo_egalizata_norm[256] = {};

	float M = img.rows * img.cols;

	histo_egalizata_norm[0] = h[0];
	for (int i = 1; i < 256; i++) {
		histo_egalizata_norm[i] = histo_egalizata_norm[i - 1] + h[i];
	}
	for (int i = 0; i < 256; i++) {
		histo_egalizata_norm[i] = histo_egalizata_norm[i] / M;
	}

	int histo_egalizata[256] = {};
	for (int i = 0; i < 256; i++) {
		histo_egalizata[i] = 255 * histo_egalizata_norm[i];
	}

	Mat dest = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dest.at<uchar>(i, j) = histo_egalizata[img.at<uchar>(i, j)];
		}
	}

	int* new_histo = histo(dest);
	imshow("resultat", dest);
	showHistogram("histo", h, 256, 256);
	showHistogram("new histo", new_histo, 256, 256);
	waitKey();
}

// LAB 9

void filtru_aritmetic() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int filter[3][3] = { 1,1,1,1,1,1,1,1,1 };
	float scalar = 9;

	Mat result = img.clone();

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			int pixel = 0;

			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					pixel += filter[x][y] * img.at<uchar>(i + x - 1, j + y - 1);
				}
			}

			result.at<uchar>(i, j) = pixel / scalar;
		}
	}

	imshow("filtru-aritmetic", result);
	waitKey(0);
}


void filtru_gaussian() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int filter[3][3] = { 1,2,1,2,4,2,1,2,1 };
	float scalar = 16;

	Mat result = img.clone();

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			int pixel = 0;

			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					pixel += filter[x][y] * img.at<uchar>(i + x - 1, j + y - 1);
				}
			}

			result.at<uchar>(i, j) = pixel / scalar;
		}
	}

	imshow("filtru-gaussian", result);
	waitKey(0);
}

void general_filter() {
	short vals[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	Mat kernel(3, 3, CV_16SC1, vals);
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	int sum = 0;
	float scalar;
	int s_positive = 0, s_negative = 0;
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			sum += kernel.at<short>(i, j);
			if (kernel.at<short>(i, j) > 0)
				s_positive += kernel.at<short>(i, j);
			else
				s_negative -= kernel.at<short>(i, j);
		}
	}

	if (s_negative == 0) 
	{// low filter only positive values
		scalar = 1.0 / sum;
	}
	else 
	{ // high filter
		scalar = 1.0 / (2 * max(s_positive, s_negative));
	}

	int half = kernel.rows / 2;

	Mat res = img.clone();

	for (int i = half; i < img.rows - half; i++) {
		for (int j = half; j < img.cols - half; j++) {
			float conv = 0;
			for (int y = 0; y < kernel.rows; y++) {
				for (int z = 0; z < kernel.cols; z++) {
					uchar pixel = img.at<uchar>(i + y - half, j + z - half);
					conv += kernel.at<short>(y, z) * pixel;
				}
			}

			conv *= scalar;

			if (s_negative != 0) { // high pass filter
				conv += (255 / 2);
			}

			conv = max(0, conv);
			conv = min(255, conv);
			res.at<uchar>(i, j) = conv;
		}
	}

	imshow("filtru", res);
	waitKey(0);
}

//LAB 10

void swap(int* a, int* b) {
	int aux = *a;
	*a = *b;
	*b = aux;
}

void bubbleSort(int v[], int n) {
	int swapped = 1;
	for (int i = 0; i < n - 1 && swapped == 1; i++) {
		swapped = 0;
		for (int j = 0; j < n - i - 1; j++) {
			if (v[j] > v[j + 1]) {
				swap(&v[j], &v[j + 1]);
				swapped = 1;
			}
		}
	}
}

void filtru_median(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	Mat result = img.clone();

	int* values = (int*)calloc(w * w, sizeof(int));
	int half = w / 2;

	for (int i = half; i < img.rows - half; i++) 
	{
		for (int j = half; j < img.cols - half; j++) 
		{
			int index = 0;
			for (int x = 0; x < w; x++)
			{
				for (int y = 0; y < w; y++) 
				{
					values[index] = img.at<uchar>(i + x - half, j + y - half);
					index++;
				}
			}

			bubbleSort(values, w * w);
			result.at<uchar>(i, j) = values[(w * w) / 2];
		}
	}

	imshow("filtru-median", result);
	waitKey();
}

void filtru_gaussian_bidimensional(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", img);

	float deviatia_standard = w / 6.0;


	Mat kernel = Mat(w, w, CV_32FC1);

	for (int i = 0; i < kernel.rows; i++)
	{
		for (int j = 0; j < kernel.cols; j++)
		{
			float deviatia_standard_patrat = pow(deviatia_standard, 2);
			float value = 1 / CV_PI / deviatia_standard_patrat;

			value *= pow(2.71, -(pow(i - kernel.rows / 2, 2) + pow(j - kernel.cols / 2, 2)) / 2 / deviatia_standard_patrat);
		}
	}
}

//LAB 11



int main()
{
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	int* h;
	int* h2;

	Mat img;
	Mat img2;

	int op;
	do
	{
		//system("cls");
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
		printf(" 10 - Additive factor\n");
		printf(" 11 - Multiply factor\n");
		printf(" 12 - Create image\n");
		printf("13 - Inverse Matrix\n");
		printf(" 14 - Split image\n");
		printf(" 15 - From color to grayscale\n");
		printf(" 16 - From grayscale to binary\n");
		printf(" 17 - From RGB to HSV\n");
		printf("18 - Is Inside\n");
		printf("19 - Show Histogram\n");
		printf("20 - Reduced Histogram\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

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
			additiveFactor(100);
			break;
		case 11:
			multiplyFactor(1);
			break;
		case 12:
			createImage();
			break;
		case 13:
			inverseMatrix();
			break;
		case 14:
			splitImage();
			break;
		case 15:
			fromColorToGrayscale();
			break;
		case 16:
			fromGrescaleToBinary(130);
			break;
		case 17:
			fromRGBtoHSV();
			break;
		case 18:
			//isInside();
			break;
		case 19:
			img = readImage(fname, 1);
			h = histo(img);

			showHistogram("HISTOGRAM", h, 256, 256);
			waitKey(0);
			break;
		case 20:
			img2 = readImage(fname2, 1);
			h2 = reduct_histo(img2, 100);

			showHistogram("REDUCED HISTOGRAM", h2, 256, 256);
			waitKey(0);
			break;
		case 23:
			traversare_in_latime();
			break;
		case 24:
			etichetare_clase_de_echivalenta();
			break;
		case 25:
			contur();
			break;
		case 26:
			reconstruct_contur_from_file();
			break;
		case 30	:
			//info_histo();
			break;
		case 50:
			//filtru_median();
			break;
		case 51:
			//filtru_gaussian_bidimensional();
			break;
		case 0:
			break;
		}

	} while (op != 0);
	return 0;
}



