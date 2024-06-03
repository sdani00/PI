// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

void binary(Mat img, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

Mat open(Mat img) {
	int di[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	int dj[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

	Mat dst1 = img.clone();

	//eroziune 
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (img.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					if (img.at<uchar>(i + di[k], j + dj[k]) == 255) {
						dst1.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	Mat dst2 = dst1.clone();

	//dilatare
	for (int i = 1; i < dst1.rows - 1; i++) {
		for (int j = 1; j < dst1.cols - 1; j++) {
			if (dst1.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					dst2.at<uchar>(i + di[k], j + dj[k]) = 0;
				}
			}
		}
	}

	return dst2;
}

Mat dilate_white(Mat img) {
	uchar val;

	int di[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	int dj[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

	Mat dst = img.clone();

	//dilatare
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (img.at<uchar>(i, j) == 255) {
				for (int k = 0; k < 8; k++) {
					dst.at<uchar>(i + di[k], j + dj[k]) = 255;
				}
			}
		}
	}

	return dst;
}

void negative(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
}

Mat subtract(Mat from, Mat toSubtract) {
	Mat notToSubtract = toSubtract.clone();
	negative(notToSubtract);

	Mat result = Mat(from.rows, from.cols, CV_8UC1);
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			if (from.at<uchar>(i, j) == 0 && notToSubtract.at<uchar>(i, j) == 0) {
				result.at<uchar>(i, j) = 0;
			}
			else {
				result.at<uchar>(i, j) = 255;
			}
		}
	}

	return result;
}

Mat erode(Mat img) {
	uchar val;
	int di[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	int dj[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };


	Mat dst = img.clone();

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (img.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					if (img.at<uchar>(i + di[k], j + dj[k]) == 255) {
						dst.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	return dst;
}

Mat or (Mat m1, Mat m2) {
	Mat result = Mat(m1.rows, m1.cols, CV_8UC1);

	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			if (m1.at<uchar>(i, j) == 0 || m2.at<uchar>(i, j) == 0) {
				result.at<uchar>(i, j) = 0;
			}
			else {
				result.at<uchar>(i, j) = 255;
			}
		}
	}

	return result;
}

long countZero(Mat img) {
	long count = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) {
				count++;
			}
		}
	}

	return count;
}

long countNonZero(Mat img)
{
	long count = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) != 0) {
				count++;
			}
		}
	}

	return count;
}

void sketelonize1() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);

	binary(img, 127);
	Mat original = img.clone();

	Mat skel = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < skel.rows; i++) {
		for (int j = 0; j < skel.cols; j++) {
			skel.at<uchar>(i, j) = 255;
		}
	}

	int i = 0;

	while (true) {
		Mat openS = open(img);

		Mat temp = subtract(img, openS);

		Mat eroded = erode(img);

		skel = or (skel, temp);

		img = eroded.clone();

		if (countZero(img) == 0) {
			break;
		}

		i++;
	}

	Mat result = subtract(original, skel);
	Mat op = dilate_white(result);

	imshow("result", op);
	waitKey();
}

/// <summary>
/// 
/// 
/// 
///   SKEL 2
/// 
/// 
/// 
/// </summary>

bool pixel_has_from_2_to_6_black_neighbors(Mat img, int i, int j) {
	int di[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	int dj[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

	int black_neighbors = 0;
	for (int d = 0; d < 8; d++) {
		if (img.at<uchar>(i + di[d], j + dj[d]) == 0) {
			black_neighbors++;
		}
	}

	if (2 <= black_neighbors && black_neighbors <= 6)
		return true;

	return false;
}

bool only_one_transition_from_white_to_black_neighbors(Mat img, int i, int j) {
	int di[9] = { -1, -1, 0, 1, 1,  1,  0, -1, -1 };
	int dj[9] = { 0,  1, 1, 1, 0, -1, -1, -1,  0 };

	int transitions = 0;
	for (int d = 0; d < 8; d++) {
		if (img.at<uchar>(i + di[d], j + dj[d]) == 255 && img.at<uchar>(i + di[d + 1], j + dj[d + 1]) == 0) {
			transitions++;
		}
	}

	if (transitions == 1)
		return true;

	return false;
}

bool at_least_one_of_P2_P4_P6_is_white(Mat img, int i, int j) {
	if (img.at<uchar>(i - 1, j) == 255 || img.at<uchar>(i, j + 1) == 255 || img.at<uchar>(i + 1, j) == 255)
		return true;

	return false;
}

bool at_least_one_of_P4_P6_P8_is_white(Mat img, int i, int j) {
	if (img.at<uchar>(i, j + 1) == 255 || img.at<uchar>(i + 1, j) == 255 || img.at<uchar>(i, j - 1) == 255)
		return true;

	return false;
}

bool at_least_one_of_P2_P4_P8_is_white(Mat img, int i, int j) {
	if (img.at<uchar>(i - 1, j) == 255 || img.at<uchar>(i, j + 1) == 255 || img.at<uchar>(i, j - 1) == 255)
		return true;

	return false;
}

bool at_least_one_of_P2_P6_P8_is_white(Mat img, int i, int j) {
	if (img.at<uchar>(i - 1, j) == 255 || img.at<uchar>(i + 1, j) == 255 || img.at<uchar>(i, j - 1) == 255)
		return true;

	return false;
}


void sketelonize2()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);

	binary(img, 127);
	Mat original = img.clone();


	while (true)
	{
		Mat copy = img.clone();
		// all pixels that meet the criteria are set to white
		Mat thin = img.clone();
		for (int i = 1; i < img.rows - 1; i++)
		{
			for (int j = 1; j < img.cols - 1; j++)
			{
				if (img.at<uchar>(i, j) == 0 && pixel_has_from_2_to_6_black_neighbors(img, i, j) && only_one_transition_from_white_to_black_neighbors(img, i, j))
				{
					if (at_least_one_of_P2_P4_P6_is_white(img, i, j) && at_least_one_of_P4_P6_P8_is_white(img, i, j))
					{
						thin.at<uchar>(i, j) = 255;
					}

					if (at_least_one_of_P2_P4_P8_is_white(img, i, j) && at_least_one_of_P2_P6_P8_is_white(img, i, j))
					{
						thin.at<uchar>(i, j) = 255;
					}
				}
			}
		}

		img = thin;

		bool same = true;
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) != copy.at<uchar>(i, j)) {
					same = false;
					break;
				}
			}
		}

		if (same) {
			break;
		}
	}

	Mat result = subtract(original, img);

	imshow("result", result);
	waitKey();
}

/// <summary>
/// 
/// 
/// 
///   SKEL 3
/// 
/// 
/// 
/// </summary>

void sub_iteration1(Mat img, int iteration)
{
	Mat marker = Mat::zeros(img.size(), CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{

				if (iteration == 1)
				{
					if (pixel_has_from_2_to_6_black_neighbors(img, i, j)
						&& only_one_transition_from_white_to_black_neighbors(img, i, j)
						&& at_least_one_of_P2_P4_P6_is_white(img, i, j)
						&& at_least_one_of_P4_P6_P8_is_white(img, i, j))
					{
						marker.at<uchar>(i, j) = 255;
					}
				}
				else
				{
					if (pixel_has_from_2_to_6_black_neighbors(img, i, j)
						&& only_one_transition_from_white_to_black_neighbors(img, i, j)
						&& at_least_one_of_P2_P4_P8_is_white(img, i, j)
						&& at_least_one_of_P2_P6_P8_is_white(img, i, j))
					{
						marker.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (marker.at<uchar>(i, j) == 255)
			{
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void guo_hall_thinning(Mat img)
{
	img /= 255;

	Mat prev = Mat::zeros(img.size(), CV_8UC1);
	Mat diff;

	do
	{
		sub_iteration1(img, 1);
		sub_iteration1(img, 2);

		subtract(prev, img, diff);
		img.copyTo(prev);
	} while (countNonZero(diff) > 0);

	img *= 255;
}

void skeletonize3()
{
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat img = imread(fname, IMREAD_GRAYSCALE);

	binary(img, 127);
	guo_hall_thinning(img);

	Mat original = img.clone();

	imshow("original", original);
	imshow("result", img);

	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 -  Skeletonize\n");
		printf(" 2 -  Skeletonize\n");
		printf(" 3 -  Skeletonize\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			sketelonize1();
			break;
		case 2:
			sketelonize2();
			break;
		case 3:
			skeletonize3();
			break;
		}

	} while (op != 0);

	return 0;
}