#pragma once
#include <opencv2/opencv.hpp>
#include "pch.h"

#include <opencv2/opencv.hpp>

cv::Mat read_frame(FILE* process, int width, int height);
void write_frame(FILE* output_pipe, const cv::Mat& frame);
void run_ffmpeg(const std::string& cmd, const std::string& mode, std::function<void(HANDLE)> process_frame);
FILE* handle_to_file(HANDLE handle);
