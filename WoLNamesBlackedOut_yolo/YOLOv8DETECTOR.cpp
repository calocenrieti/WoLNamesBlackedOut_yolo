#include <memory>
#include <iostream>
#include <Windows.h>
#include "pch.h"
#include <optional>
#include <vector>
#include <fstream>
#include <winrt/Windows.Storage.h>
#include <queue>
#include <mutex>
#include <thread>

#define MY_MODULE_EXPORTS
#ifdef MY_MODULE_EXPORTS
#define MY_API __declspec(dllexport)
#else
#define MY_API __declspec(dllimport)
#endif

class MY_API YOLOv8Detector {
public:
    struct BoundingBox {
        int index;
        float score;
        cv::Rect rect; // OpenCVのRectに変更
    };
private:
    struct Private;
    std::unique_ptr<Private> m; // ポインタをstd::unique_ptrに変更

    std::mutex mutex_; // std::mutex型で宣言
public:
    YOLOv8Detector();
    ~YOLOv8Detector();
    operator bool() const;
    bool loadModel(const char* model_path);
    std::optional<std::vector<BoundingBox>> inference(const cv::Mat& image);
    bool PreProcess(cv::Mat& iImg, int iImgSize, cv::Mat& oImg);
    float resizeScales;
};

struct YOLOv8Detector::Private {

    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "YOLOv8Detector" };
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;

    size_t num_input_nodes = 0;
    size_t num_output_nodes = 0;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<std::string> input_node_name_strings;
    std::vector<std::string> output_node_name_strings;

};


YOLOv8Detector::YOLOv8Detector()
    : m(std::make_unique<Private>()) // std::make_uniqueを使用
{
    try {
        //std::cout << "Initializing memory_info..." << std::endl;
        m->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        //std::cout << "memory_info initialized." << std::endl;

        Ort::GetApi().SessionOptionsAppendExecutionProvider(m->session_options, "DML", nullptr, nullptr, 0);

        m->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m->session_options.SetLogSeverityLevel(3);

        m->session_options.SetIntraOpNumThreads(1);
        //std::cout << "Session options configured." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
    }
}



YOLOv8Detector::~YOLOv8Detector() = default;

YOLOv8Detector::operator bool() const
{
    return m->session != nullptr;
}

bool YOLOv8Detector::loadModel(const char* model_path)
{
    try {
        //std::cout << "Loading model..." << std::endl;

        // UTF-8からUTF-16 (wchar_t) に変換
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
        std::wstring w_model_path(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, &w_model_path[0], size_needed);

        //std::cout << "Creating session..." << std::endl;
        m->session = std::make_unique<Ort::Session>(m->env, w_model_path.c_str(), m->session_options);

        //std::cout << "Session created." << std::endl;

        m->num_input_nodes = m->session->GetInputCount();
        m->num_output_nodes = m->session->GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        m->input_node_names.resize(m->num_input_nodes);
        m->input_node_name_strings.resize(m->num_input_nodes);
        for (size_t i = 0; i < m->num_input_nodes; i++) {
            auto input_name = m->session->GetInputNameAllocated(i, allocator);
            m->input_node_name_strings[i] = input_name.get();
            m->input_node_names[i] = m->input_node_name_strings[i].c_str();
        }

        m->output_node_names.resize(m->num_output_nodes);
        m->output_node_name_strings.resize(m->num_output_nodes);
        for (size_t i = 0; i < m->num_output_nodes; i++) {
            auto output_name = m->session->GetOutputNameAllocated(i, allocator);
            m->output_node_name_strings[i] = output_name.get();
            m->output_node_names[i] = m->output_node_name_strings[i].c_str();
        }

        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return false;
}




bool YOLOv8Detector::PreProcess(cv::Mat& iImg, int iImgSize, cv::Mat& oImg)
{
    oImg = iImg.clone();
    
    if (iImg.cols >= iImg.rows)
    {
        resizeScales = iImg.cols / (float)iImgSize;
        cv::resize(oImg, oImg, cv::Size(iImgSize, int(iImg.rows / resizeScales)));
    }
    else
    {
        resizeScales = iImg.rows / (float)iImgSize;
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize));
    }
    cv::Mat tempImg = cv::Mat::zeros(iImgSize, iImgSize, CV_8UC3);
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;
    
    //cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    return true;
}


std::optional<std::vector<YOLOv8Detector::BoundingBox>> YOLOv8Detector::inference(const cv::Mat& image)
{
    const int N = 1; // batch size
    const int C = 3; // number of channels
    const int W = 1280; // width
    const int H = 1280; // height

    std::vector<float> input_tensor_values(N * C * H * W);

    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0); // 正規化

    float* R = input_tensor_values.data() + H * W * 0;
    float* G = input_tensor_values.data() + H * W * 1;
    float* B = input_tensor_values.data() + H * W * 2;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            cv::Vec3f pixel = resized_img.at<cv::Vec3f>(y, x);
            R[W * y + x] = pixel[2];
            G[W * y + x] = pixel[1];
            B[W * y + x] = pixel[0];
        }
    }

    std::vector<int64_t> input_tensor_shape = { N, C, H, W };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(*m->memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

    auto output_tensors = m->session->Run(Ort::RunOptions{ nullptr }, m->input_node_names.data(), &input_tensor, 1, m->output_node_names.data(), m->num_output_nodes);

    std::vector<YOLOv8Detector::BoundingBox> bboxes;

    auto info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();

    assert(shape.size() == 3);
    assert(shape[0] == N);
    assert(shape[1] > 4);

    const int values = shape[1];    //x,y,w,h,class1,class2,    84
    const int classes = values - 4; //項目からx,y,w,hを引いた数  80
    const int count = shape[2];	    //グリッド数    8400

    float const* output_tensor = output_tensors[0].GetTensorData<float>();
    float conf_threshold = 0.01;
    float iou_threshold = 0.5;

    for (int i = 0; i < count; i++) {
        auto Value = [&](int index) {
            return output_tensor[count * index + i];
            };

        float x = Value(0) * image.cols / W;
        float y = Value(1) * image.rows / H;
        float w = Value(2) * image.cols / W;
        float h = Value(3) * image.rows / H;

        x = int((x - w / 2) * resizeScales);
        y = int((y - h / 2) * resizeScales);
        w = int(w * resizeScales);
        h = int(h * resizeScales);

        for (int j = 0; j < classes; j++) {
            YOLOv8Detector::BoundingBox bbox;
            bbox.score = Value(4 + j);

            if (bbox.score > 0.01) {
                bbox.index = j;
                bbox.rect = cv::Rect(x, y, w, h);
                bboxes.push_back(bbox);
            }
        }
    }

    return bboxes;
}


// 矩形情報の構造体を定義
struct RectInfo {
    int x;
    int y;
    int width;
    int height;
};

// 色情報の構造体を定義
struct ColorInfo {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// フレームカウントのグローバル変数
extern "C" __declspec(dllexport) int total_frame_count = 0;
// 必要に応じて初期化コードを追加
extern "C" __declspec(dllexport) int get_total_frame_count() {
    return total_frame_count;
}

cv::Mat put_C_SQUARE_ENIX(cv::Mat img, cv::Mat c_sqex_image, int w, int h)
{
    // 貼り付け位置を計算
    int dy = h - 41;
    int dx = w - 220;

    // 貼り付ける画像のサイズを取得
    int image_h = c_sqex_image.rows;
    int image_w = c_sqex_image.cols;

    // 画像の部分領域を定義
    cv::Rect roi(dx, dy, image_w, image_h);

    // 貼り付ける画像をROI（領域）にコピー
    c_sqex_image.copyTo(img(roi));

    return img;
}

// preview_api関数を宣言
extern "C" __declspec(dllexport) MY_API int preview_api(const char* image_path,  RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color,bool inpaint,bool copyright, bool no_inference)
{
	std::string imagePathStr(image_path);
    // モデルファイルのパス
    const char* model_path = "./my_yolov8m.onnx";
    // 画像ファイルのパス
    //const char* image_path = "./b_output_0015.png";

    // OpenCV の色を作成
    cv::Scalar name_color_Scalar(name_color.b, name_color.g, name_color.r);
	cv::Scalar fixframe_Scalar(fixframe_color.b, fixframe_color.g, fixframe_color.r);

    // YOLOv8Detectorのインスタンスを作成
    YOLOv8Detector detector;


    // モデルをロード
    if (!detector.loadModel(model_path)) {
        //std::cerr << "Failed to load model." << std::endl;
        return -1;
    }
    // 画像を読み込み
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        //std::cerr << "Failed to load image." << std::endl;
        return -1;
    }
    cv::Mat o_image = image.clone();

    if (no_inference==true)
	{
		// 画像の前処理
		detector.PreProcess(image, 1280, o_image);
		// 物体検出
		auto result = detector.inference(o_image);
		if (result) {
            // NMS (非最大抑制) を適用
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            for (const auto& bbox : result.value()) {
                boxes.push_back(bbox.rect);
                scores.push_back(bbox.score);
            }

            std::vector<int> indices;

            float scoreThreshold = 0.01f; // スコアのしきい値（適宜変更）
            float nmsThreshold = 0.5f;   // NMS のしきい値（適宜変更）
            cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
			if (inpaint == true)
			{
                cv::Mat mask_frame = cv::Mat::zeros(image.size(), CV_8UC1);

                for (const auto& i : indices) {
                    // 矩形を白（255）で塗りつぶす
                    cv::rectangle(mask_frame, boxes[i], cv::Scalar(255), cv::FILLED);

				}
				cv::Mat inpainted_image = image.clone();
                cv::inpaint(image, mask_frame, inpainted_image, 3, cv::INPAINT_TELEA);
                image = inpainted_image;

			}
			else
			{
                // 検出結果をBBOXで描画
                for (const auto& i : indices) {
                    cv::rectangle(image, boxes[i], name_color_Scalar, -1);
                }
			}
		}
	}

    // 矩形を描画
    for (int i = 0; i < count; i++)
    {
        cv::Rect rect(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
        cv::rectangle(image, rect, fixframe_Scalar, -1);
    }

	if (copyright == true)
	{
		// 貼り付ける画像を読み込む
		cv::Mat c_sqex_image = cv::imread("C_SQUARE_ENIX.png");
		// 画像の幅と高さ
		int w = image.cols;
		int h = image.rows;
		// 関数を呼び出して画像を貼り付け
		image = put_C_SQUARE_ENIX(image, c_sqex_image, w, h);
	}
    // 画像を書き込み
    cv::imwrite(image_path, image);
 

    return 0;
}



void write_frame(FILE* output_pipe, const cv::Mat& frame) {
    fwrite(frame.data, 1, frame.total() * frame.elemSize(), output_pipe);
}


void dml_process_frame(const cv::Mat& in_frame, cv::Mat& out_frame, YOLOv8Detector& detector, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool inpaint, bool copyright, bool no_inference)
{
    // OpenCV の色を作成
    cv::Scalar fixframe_Scalar(fixframe_color.b, fixframe_color.g, fixframe_color.r);
    out_frame = in_frame.clone();

    std::cerr << "dml_process_frame: Start processing frame" << std::endl;

    if (no_inference == true)
    {
        cv::Scalar name_color_Scalar(name_color.b, name_color.g, name_color.r);

        cv::Mat processed_frame;
        detector.PreProcess(const_cast<cv::Mat&>(in_frame), in_frame.cols, processed_frame);

        auto result = detector.inference(processed_frame);

        if (result) {
            // NMS (非最大抑制) を適用
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            for (const auto& bbox : result.value()) {
                boxes.push_back(bbox.rect);
                scores.push_back(bbox.score);
            }

            std::vector<int> indices;

            float scoreThreshold = 0.01f; // スコアのしきい値（適宜変更）
            float nmsThreshold = 0.5f;   // NMS のしきい値（適宜変更）
            cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
            if (inpaint == true)
            {
                cv::Mat mask_frame = cv::Mat::zeros(in_frame.size(), CV_8UC1);

                for (const auto& i : indices) {
                    // 矩形を白（255）で塗りつぶす
                    cv::rectangle(mask_frame, boxes[i], cv::Scalar(255), cv::FILLED);
                }
                cv::Mat inpainted_image = in_frame.clone();
                cv::inpaint(in_frame, mask_frame, inpainted_image, 3, cv::INPAINT_TELEA);
                out_frame = inpainted_image;
            }
            else
            {
                // 検出結果をBBOXで描画
                for (const auto& i : indices) {
                    cv::rectangle(out_frame, boxes[i], name_color_Scalar, -1);
                }
            }
        }
    }
    // 矩形を描画
    for (int i = 0; i < count; i++)
    {
        cv::Rect rect(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
        cv::rectangle(out_frame, rect, fixframe_Scalar, -1);
    }

    if (copyright == true)
    {
        // 貼り付ける画像を読み込む
        cv::Mat c_sqex_image = cv::imread("C_SQUARE_ENIX.png");
        if (!c_sqex_image.empty()) {
            // 画像の幅と高さ
            int w = out_frame.cols;
            int h = out_frame.rows;
            // 関数を呼び出して画像を貼り付け
            out_frame = put_C_SQUARE_ENIX(out_frame, c_sqex_image, w, h);
        }
    }

    std::cerr << "dml_process_frame: Finished processing frame" << std::endl;
}



cv::Mat read_frame(FILE* process, int width, int height) {
    std::cerr << "Before buffer allocation - Width: " << width << ", Height: " << height << std::endl;
    cv::Mat frame(height, width, CV_8UC3);
    size_t frame_size = width * height * 3;
    std::cerr << "Frame size: " << frame_size << std::endl;
    std::vector<unsigned char> buffer(frame_size);
    std::cerr << "After buffer allocation - Width: " << width << ", Height: " << height << std::endl;

    size_t bytes_read = fread(buffer.data(), 1, frame_size, process);
    std::cerr << "fread returned bytes_read: " << bytes_read << std::endl; // ★ fread の戻り値をログ出力
    std::cerr << "feof(process) is: " << feof(process) << std::endl;    // ★ feof の評価結果をログ出力
    std::cerr << "ferror(process) is: " << ferror(process) << std::endl;  // ★ ferror の評価結果をログ出力

    if (bytes_read != frame_size) {
        std::cerr << "Expected to read " << frame_size << " bytes, but only read " << bytes_read << " bytes." << std::endl;
        if (feof(process)) {
            std::cerr << "Error: Reached end of file before reading expected frame size." << std::endl;
            return cv::Mat(); // ★ EOF の場合は空の cv::Mat を返す

        }
        else if (ferror(process)) {
            std::cerr << "Error: An error occurred while reading the file." << std::endl;
            return cv::Mat(); // エラー発生時も空の cv::Mat を返す (処理を継続するため)
        }
        else {
            std::cerr << "Error: Unknown error occurred while reading the file." << std::endl;
            return cv::Mat(); // エラー発生時も空の cv::Mat を返す (処理を継続するため)
        }
        // throw std::runtime_error("Failed to read frame data"); // 例外throwは削除
        return cv::Mat(); // エラー発生時も空の cv::Mat を返す (処理を継続するため)
    }

    std::memcpy(frame.data, buffer.data(), frame_size);
    return frame;
}

void run_ffmpeg(const char* cmd, const char* mode, std::function<void(FILE*)> func) {
    FILE* pipe = _popen(cmd, mode);
    if (!pipe) {
        std::cerr << "Failed to run ffmpeg command" << std::endl;
        return;
    }
    func(pipe);
    _pclose(pipe);
}

//DirectMLを使用した物体検出処理
extern "C" __declspec(dllexport) MY_API int dml_main(char* input_video_path, char* output_video_path, char* codec, char* hwaccel, int width, int height, int fps,char* color_primaries,  RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool inpaint, bool copyright, bool no_inference)
{
    const char* model_path = "./my_yolov8m.onnx";

    YOLOv8Detector detector;


    if (!detector.loadModel(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    
    std::string ffmpeg_input_cmd;
    if (std::string(color_primaries) !="bt709") {
        ffmpeg_input_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-vf \"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }
    else {
        ffmpeg_input_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }

    std::string ffmpeg_output_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) + " -i - -movflags faststart -pix_fmt yuv420p -vcodec " + std::string(codec) +
        " -b:v 11M -preset slow \"" + std::string(output_video_path) + "\"";

    cv::Mat current_frame;
    cv::Mat processed_frame;
    total_frame_count = 0;

    // フレームを保存するキュー
    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool finished_reading = false;
    const int max_queue_size = 100; // 適宜設定

    // 読み取りスレッド
    auto read_thread = std::thread([&]() {
        run_ffmpeg(ffmpeg_input_cmd.c_str(), "rb", [&](FILE* input_pipe) {
            while (true) {
                current_frame = read_frame(input_pipe, width, height);
                if (current_frame.empty()) {
                    break;
                }
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&]() { return frame_queue.size() < max_queue_size; });
                    frame_queue.push(current_frame);
                }
                queue_cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                finished_reading = true;
            }
            queue_cv.notify_one();
            });
        });

    auto write_thread = std::thread([&]() {
        run_ffmpeg(ffmpeg_output_cmd.c_str(), "wb", [&](FILE* output_pipe) {
            if (!output_pipe) {
                std::cerr << "Error: output_pipe is nullptr in write_thread" << std::endl;
                return;
            }
            while (true) {
                cv::Mat processed_frame;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&]() { return !frame_queue.empty() || finished_reading; });

                    if (frame_queue.empty() && finished_reading) {
                        break;
                    }

                    processed_frame = frame_queue.front();
                    frame_queue.pop();
                }
                queue_cv.notify_one();

                if (!processed_frame.empty()) {
                    dml_process_frame(processed_frame, processed_frame, detector, rects, count, name_color, fixframe_color, inpaint, copyright, no_inference);
                    total_frame_count += 1;
                    write_frame(output_pipe, processed_frame);
                }
            }

            queue_cv.notify_one();
            });
        });

    read_thread.join();
    write_thread.join();

    return 0;
}




std::vector<unsigned char> load_engine_file(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}

void LetterBox(const cv::Mat& image, cv::Mat& outImage,
    cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
    const cv::Size& newShape = cv::Size(1280, 1280),
    bool autoShape = false,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(0, 0, 0)
);



void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
    bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

//void postprocess(float(&rst)[1][5][33600], cv::Mat& img, cv::Vec4d params);
void postprocess(float* rst, int batch_size, std::vector<cv::Mat>& images, std::vector<cv::Vec4d>& params, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool inpaint, bool copyright, bool no_inference)
{
    // OpenCV の色を作成

    cv::Scalar fixframe_Scalar(fixframe_color.b, fixframe_color.g, fixframe_color.r);

    for (int b = 0; b < batch_size; ++b) {
        if (no_inference == true)
        {
            cv::Scalar name_color_Scalar(name_color.b, name_color.g, name_color.r);

            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> det_rst;
            static const float score_threshold = 0.01;
            static const float nms_threshold = 0.5;
            std::vector<int> indices;

            for (int Anchors = 0; Anchors < 33600; Anchors++)
            {
                float max_score = 0.0;
                int max_score_det = 99;
                float pdata[4];
                int prob = 4;
                {
                    if (rst[b * 5 * 33600 + prob * 33600 + Anchors] > max_score) {
                        max_score = rst[b * 5 * 33600 + prob * 33600 + Anchors];
                        max_score_det = prob - 4;
                        pdata[0] = rst[b * 5 * 33600 + 0 * 33600 + Anchors];
                        pdata[1] = rst[b * 5 * 33600 + 1 * 33600 + Anchors];
                        pdata[2] = rst[b * 5 * 33600 + 2 * 33600 + Anchors];
                        pdata[3] = rst[b * 5 * 33600 + 3 * 33600 + Anchors];
                    }
                }
                if (max_score >= score_threshold)
                {
                    float x = (pdata[0] - params[b][2]) / params[b][0];
                    float y = (pdata[1] - params[b][3]) / params[b][1];
                    float w = pdata[2] / params[b][0];
                    float h = pdata[3] / params[b][1];
                    int left = MAX(int(x - 0.5 * w + 0.5), 0);
                    int top = MAX(int(y - 0.5 * h + 0.5), 0);
                    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
                    scores.emplace_back(max_score);
                    det_rst.emplace_back(max_score_det);
                }
            }

            cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);

            if (inpaint == true)
            {
                cv::Mat mask_frame = cv::Mat::zeros(images[b].size(), CV_8UC1);

                for (const auto& i : indices) {
                    // 矩形を白（255）で塗りつぶす
                    cv::rectangle(mask_frame, boxes[i], cv::Scalar(255), cv::FILLED);
                }
                cv::Mat inpainted_image = images[b].clone();
                cv::inpaint(images[b], mask_frame, inpainted_image, 3, cv::INPAINT_TELEA);
                images[b] = inpainted_image;
            }
            else
            {

                for (int i = 0; i < indices.size(); i++) {
                    //cv::rectangle(images[b], boxes[indices[i]], cv::Scalar(0, 0, 0), -1, cv::LINE_8, 0);
                    cv::rectangle(images[b], boxes[indices[i]], name_color_Scalar, -1);
                }
            }

        }

        // 矩形を描画
        for (int i = 0; i < count; i++)
        {
            cv::Rect rect(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
            cv::rectangle(images[b], rect, fixframe_Scalar, -1);
        }

        if (copyright == true)
        {
            // 貼り付ける画像を読み込む
            cv::Mat c_sqex_image = cv::imread("C_SQUARE_ENIX.png");
            if (!c_sqex_image.empty()) {
                // 画像の幅と高さ
                int w = images[b].cols;
                int h = images[b].rows;
                // 関数を呼び出して画像を貼り付け
                images[b] = put_C_SQUARE_ENIX(images[b], c_sqex_image, w, h);
            }
        }
    }

}


extern "C" __declspec(dllexport) MY_API int trt_main(char* input_video_path, char* output_video_path, char* codec, char* hwaccel, int width, int height, int fps, char* color_primaries, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool inpaint, bool copyright, bool no_inference)
{
    using namespace winrt::Windows::Storage;

    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    };
    Logger logger;

    std::unique_ptr<nvinfer1::IRuntime> runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (runtime == nullptr) { return false; }

    const int batch_size = 16; // バッチサイズを設定
    const size_t max_queue_size = 100; // キューの最大サイズを設定

    // ローカルフォルダのパスを取得
    auto localFolder = ApplicationData::Current().LocalFolder();
    std::wstring localAppDataPath = localFolder.Path().c_str();

    // アプリケーション専用のフォルダパスを組み立てる
    std::wstring appFolderPath = std::wstring(localAppDataPath) + std::wstring{ L"\\WoLNamesBlackedOut" };
    std::wstring engineFilePath = appFolderPath + std::wstring{ L"\\my_yolov8m.engine" };

    std::string engineFilePathStr(engineFilePath.length(), 0);
    std::transform(engineFilePath.begin(), engineFilePath.end(), engineFilePathStr.begin(), [](wchar_t c) {
        return (char)c;
        });
    const char* engineFilePathCStr = engineFilePathStr.c_str();

    //std::string file_path = "my_yolov8m.engine";
    auto plan = load_engine_file(engineFilePathCStr);

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (engine == nullptr) { return false; }

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context == nullptr) { return false; }

    auto idims = engine->getTensorShape("images");
    auto odims = engine->getTensorShape("output0");
    nvinfer1::Dims4 inputDims = { batch_size, idims.d[1], idims.d[2], idims.d[3] };
    nvinfer1::Dims3 outputDims = { batch_size, idims.d[1], idims.d[2] };
    context->setInputShape("images", inputDims);

    void* buffers[2];
    const int inputIndex = 0;
    const int outputIndex = 1;

    cudaMalloc(&buffers[inputIndex], batch_size * idims.d[1] * idims.d[2] * idims.d[3] * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batch_size * odims.d[1] * odims.d[2] * sizeof(float));

    context->setTensorAddress("images", buffers[inputIndex]);
    context->setTensorAddress("output0", buffers[outputIndex]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    std::string ffmpeg_input_cmd;
    if (std::string(color_primaries) != "bt709") {
        ffmpeg_input_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-vf \"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }
    else {
        ffmpeg_input_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }

    std::string ffmpeg_output_cmd = ".\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe -loglevel quiet -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) + " -i - -movflags faststart -pix_fmt yuv420p -vcodec " + std::string(codec) +
        " -b:v 11M -preset slow \"" + std::string(output_video_path) + "\"";

    cv::Mat current_frame;
    cv::Mat processed_frame;
    total_frame_count = 0;

    float* rst = new float[batch_size * 5 * 33600];

    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool finished_reading = false;




    auto read_thread = std::thread([&]() {
        run_ffmpeg(ffmpeg_input_cmd.c_str(), "rb", [&](FILE* input_pipe) {
            while (true) {
                current_frame = read_frame(input_pipe, width, height);
                if (current_frame.empty()) {
                    break;
                }
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&]() { return frame_queue.size() < max_queue_size; });
                    frame_queue.push(current_frame);
                }
                queue_cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                finished_reading = true;
            }
            queue_cv.notify_one();
            });
        });

    auto write_thread = std::thread([&]() {
        run_ffmpeg(ffmpeg_output_cmd.c_str(), "wb", [&](FILE* output_pipe) {
            while (true) {
                std::vector<cv::Mat> frames;
                std::vector<cv::Vec4d> params;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&]() { return frame_queue.size() >= batch_size || (finished_reading && !frame_queue.empty()); });
                    if (frame_queue.empty() && finished_reading) {
                        break;
                    }
                    while (!frame_queue.empty() && frames.size() < batch_size) {
                        frames.push_back(frame_queue.front());
                        frame_queue.pop();
                    }
                }
                if (frames.empty()) {
                    break;
                }
                if (frame_queue.empty() && finished_reading) {
                    break;
                }
                std::vector<cv::Mat> blobs;
                for (auto& frame : frames) {
                    cv::Mat LetterBoxImg;
                    cv::Vec4d param;
                    LetterBox(frame, LetterBoxImg, param, cv::Size(1280, 1280));
                    params.push_back(param);

                    cv::Mat blob;
                    cv::dnn::blobFromImage(LetterBoxImg, blob, 1 / 255.0, cv::Size(1280, 1280), cv::Scalar(0, 0, 0), true, false, CV_32F);
                    blobs.push_back(blob);
                }

                for (int i = 0; i < blobs.size(); ++i) {
                    cudaMemcpyAsync(static_cast<float*>(buffers[inputIndex]) + i * 3 * 1280 * 1280, blobs[i].data, 3 * 1280 * 1280 * sizeof(float), cudaMemcpyHostToDevice, stream);
                }
                context->setOptimizationProfileAsync(0, stream);
                context->enqueueV3(stream);
                cudaStreamSynchronize(stream);

                cudaMemcpyAsync(rst, buffers[outputIndex], batch_size * 5 * 33600 * sizeof(float), cudaMemcpyDeviceToHost, stream);

                postprocess(rst, frames.size(), frames, params,  rects, count, name_color, fixframe_color, inpaint, copyright, no_inference);
                for (auto& frame : frames) {
                    write_frame(output_pipe, frame);
                }

                total_frame_count += frames.size();
                queue_cv.notify_one();
            }
            });
        });

    read_thread.join();
    write_thread.join();


    delete[] rst;
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}

