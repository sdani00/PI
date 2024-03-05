// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
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


int main()
{
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
		printf("18 - Is Inside");
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
		}
	} while (op != 0);
	return 0;
}
