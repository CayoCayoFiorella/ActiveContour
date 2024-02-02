#include <iostream>
#include "../headers/common.h"

void loadBar(int x, int n, int w)
{
  // Calculate the ratio of complete-to-incomplete.
  float ratio = x / static_cast<float>(n);
  int   c     = ratio * w;

  // Show the percentage complete.
  printf("Progress: %3d%% [", static_cast<int>(ratio * 100));

  // Show the load bar.
  for (int i = 0; i < c; i++)
    printf("=");

  for (int i = c; i < w; i++)
    printf(" ");

  // ANSI Control codes to go back to the
  // previous line and clear it.
  if (c == w)
    printf("]\n");
  else
  {
    printf("]\r");
    fflush(stdout);
  }
}

CV_EXPORTS_W void cvNeumannBoundCond(const cv::Mat& srcarr, cv::Mat& dstarr)
{
    cv::Mat src, dst;
    cv::Size size;

    int i, j;
    float* ptr_src, * ptr_dst;
    int iStep_src, iStep_dst;

    src = srcarr.clone();
    dst = dstarr.clone();

    if (src.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only 32-bit, 1-channel input images are supported");
    if (dst.type() != CV_32FC1)
        CV_Error(cv::Error::StsUnsupportedFormat, "Only 32-bit, 1-channel input images are supported");
    if (src.size() != dst.size())
        CV_Error(cv::Error::StsUnmatchedSizes, "The input images must have the same size");

    size = src.size();
    src.copyTo(dst);

    ptr_src = src.ptr<float>();
    iStep_src = src.step1();
    ptr_dst = dst.ptr<float>();
    iStep_dst = dst.step1();

    ptr_dst[0] = ptr_src[2 + iStep_src * 2];
    ptr_dst[size.width - 1] = ptr_src[size.width - 3 + iStep_src * 2];
    ptr_dst[iStep_dst * (size.height - 1)] = ptr_src[2 + iStep_src * (size.height - 3)];
    ptr_dst[size.width - 1 + iStep_dst * (size.height - 1)] = ptr_src[size.width - 3 + iStep_dst * (size.height - 3)];

    for (i = 1; i < size.width - 1; i++)
    {
        ptr_dst[i] = ptr_src[i + iStep_src * 2];
        ptr_dst[i + iStep_dst * (size.height - 1)] = ptr_src[i + iStep_src * (size.height - 3)];
    }

    for (j = 1; j < size.height - 1; j++)
    {
        ptr_dst[iStep_dst * j] = ptr_src[2 + iStep_src * j];
        ptr_dst[size.width - 1 + iStep_dst * j] = ptr_src[size.width - 3 + iStep_src * j];
    }
}
