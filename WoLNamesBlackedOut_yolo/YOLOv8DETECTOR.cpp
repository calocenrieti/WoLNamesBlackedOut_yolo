#include "pch.h"
#include <memory>
#include <iostream>
#include <Windows.h>
#include <optional>
#include <vector>
#include <fstream>
#include <Windows.h>
#include <winrt/Windows.Storage.h>
#include <queue>
#include <mutex>
#include <thread>
#include <winrt/base.h>
#include <numeric>
#define MY_MODULE_EXPORTS
#ifdef MY_MODULE_EXPORTS
#define MY_API __declspec(dllexport)
#else
#define MY_API __declspec(dllimport)
#endif

// グローバル変数
HANDLE g_ffmpegInputProcessHandle = nullptr;
HANDLE g_ffmpegOutputProcessHandle = nullptr;
std::atomic<bool> cancelFlag = false;
constexpr int MIN_VALID_RECT_SIZE = 10;
constexpr int DEFAULT_MOSAIC_FACTOR = 10;
constexpr int DEFAULT_BLUR_FACTOR = 5;
cv::Mat c_sqex_image = cv::imread("C_SQUARE_ENIX.png");

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
    __stdcall YOLOv8Detector();
    __stdcall ~YOLOv8Detector();
    operator bool() const;
    bool __stdcall loadModel(const char* model_path);
    std::optional<std::vector<BoundingBox>> __stdcall inference(const cv::Mat& image);
    bool __stdcall PreProcess2(cv::Mat& iImg, int targetWidth, int targetHeight, cv::Mat& oImg);
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

    bool use_directml = false; // DirectML を使えるかどうかのフラグ
};


YOLOv8Detector::YOLOv8Detector()
    : m(std::make_unique<Private>()) // std::make_uniqueを使用
{
    try {
        m->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

        // DirectML (DML) プロバイダを追加してみる（失敗したら CPU フォールバック）
        try {
            // 既存コードを活かして試行
            Ort::GetApi().SessionOptionsAppendExecutionProvider(m->session_options, "DML", nullptr, nullptr, 0);
            m->use_directml = true;
            //std::cout << "DirectML execution provider appended." << std::endl;
        }
        catch (const Ort::Exception&) {
            // DML が利用できない／初期化に失敗した -> CPU を使用する
            m->use_directml = false;
            //std::cerr << "DirectML not available, falling back to CPU execution provider." << std::endl;
        }

        m->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m->session_options.SetLogSeverityLevel(3);

        m->session_options.SetIntraOpNumThreads(1);
        //std::cout << "Session options configured." << std::endl;
    }
    catch (const std::exception& e) {
        //std::cerr << "Exception during initialization: " << e.what() << std::endl;
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
        // UTF-8からUTF-16 (wchar_t) に変換
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
        std::wstring w_model_path(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, &w_model_path[0], size_needed);

        // まず現在の session_options（DML がセットされている可能性あり）で試す
        try {
            m->session = std::make_unique<Ort::Session>(m->env, w_model_path.c_str(), m->session_options);
            m->use_directml = m->use_directml; // 状態はそのまま
        }
        catch (const Ort::Exception& e) {
            // DML での初期化に失敗した場合は CPU にフォールバックして再試行
            if (m->use_directml) {
                try {
                    Ort::SessionOptions cpu_options;
                    cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    cpu_options.SetLogSeverityLevel(3);
                    cpu_options.SetIntraOpNumThreads(1);

                    m->session = std::make_unique<Ort::Session>(m->env, w_model_path.c_str(), cpu_options);
                    m->use_directml = false;
                    //std::cerr << "Loaded model with CPU execution provider after DML failure." << std::endl;
                }
                catch (const Ort::Exception& e2) {
                    // CPUでも失敗したら終了
                    //std::cerr << "Failed to load model with CPU fallback: " << e2.what() << std::endl;
                    return false;
                }
            }
            else {
                // DML試行していない状態で失敗したらそのまま失敗
                //std::cerr << "Failed to load model: " << e.what() << std::endl;
                return false;
            }
        }

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
        //std::cerr << "Error: " << e.what() << std::endl;
    }
    return false;
}

bool YOLOv8Detector::PreProcess2(cv::Mat& iImg, int targetWidth, int targetHeight, cv::Mat& oImg)
{
    // 入力画像のコピーを作成
    oImg = iImg.clone();

    // 目標のアスペクト比
    float targetRatio = static_cast<float>(targetWidth) / targetHeight;
    // 入力画像のアスペクト比
    float imgRatio = static_cast<float>(iImg.cols) / iImg.rows;

    // 横長、またはアスペクト比が同じ場合：横幅基準でリサイズ
    if (imgRatio >= targetRatio) {
        // 横幅をtargetWidthに合わせるのでスケールは入力横幅 / targetWidth
        resizeScales = iImg.cols / static_cast<float>(targetWidth);
        // 高さは元サイズから同じスケールで調整
        cv::resize(oImg, oImg, cv::Size(targetWidth, static_cast<int>(iImg.rows / resizeScales)), 0, 0, cv::INTER_LINEAR);
    }
    // 縦長の場合：縦幅基準でリサイズ
    else {
        // 縦幅をtargetHeightに合わせるのでスケールは入力縦幅 / targetHeight
        resizeScales = iImg.rows / static_cast<float>(targetHeight);
        cv::resize(oImg, oImg, cv::Size(static_cast<int>(iImg.cols / resizeScales), targetHeight), 0, 0, cv::INTER_LINEAR);
    }

    // 出力テンソル（背景は0で埋める＝黒）の作成
    cv::Mat tempImg = cv::Mat::zeros(targetHeight, targetWidth, CV_8UC3);
    // リサイズ済み画像を左上にコピー（必要に応じて中央寄せやオフセットを追加可能）
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;

    return true;
}

std::optional<std::vector<YOLOv8Detector::BoundingBox>> YOLOv8Detector::inference(const cv::Mat& image)
{
    const int N = 1; // batch size
    const int C = 3; // number of channels
    const int W = 1280; // width
    const int H = 736; // height

    std::vector<float> input_tensor_values(N * C * H * W);

    cv::Mat resized_img;
    image.convertTo(resized_img, CV_32F, 1.0f / 255.0f); // 正規化

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

    // 出力形状を柔軟に扱う
    auto info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    if (shape.size() != 3) {
        return bboxes;
    }

    // 判別：yolo26 なら (N, detections, 6)、元の yolov8 形式なら (N, values, count)
    int64_t batch = shape[0];
    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];

    int64_t detections = 0;
    int64_t per_det = 0;
    // case: (N, 300, 6)
    if (dim2 == 6) {
        detections = dim1;
        per_det = dim2;
    }
    // case: (N, values, count) 旧コード風
    else if (dim1 == 6) {
        detections = dim2;
        per_det = dim1;
    }
    else {
        // フェールセーフ: treat dim1 as values, dim2 as count (既存ロジック)
        per_det = static_cast<int>(shape[1]);
        detections = static_cast<int>(shape[2]);
    }

    float const* output_tensor = output_tensors[0].GetTensorData<float>();
    float score_threshold = 0.15f;

    // バッチ0 を扱う（バッチサポートが必要なら拡張）
    const float* batch_base = output_tensor; // offset = 0 * detections * per_det

    for (int64_t i = 0; i < detections; ++i) {
        const float* det = batch_base + i * per_det;
        // 必要な要素が6であることを期待
        if (per_det < 6) continue;

        float score = det[4];
        if (score <= score_threshold) continue;

        float x1_model = det[0];
        float y1_model = det[1];
        float x2_model = det[2];
        float y2_model = det[3];
        int class_id = static_cast<int>(det[5]);

        // モデル座標 -> 元画像座標
        // PreProcess2 で resizeScales = original_size / targetSize としているため、
        // 元座標 = model座標 * resizeScales
        float x1 = x1_model * resizeScales;
        float y1 = y1_model * resizeScales;
        float x2 = x2_model * resizeScales;
        float y2 = y2_model * resizeScales;

        // 左上・幅高さを正しく計算
        int left = std::max(0, static_cast<int>(std::round(std::min(x1, x2))));
        int top = std::max(0, static_cast<int>(std::round(std::min(y1, y2))));
        int width = static_cast<int>(std::round(std::abs(x2 - x1)));
        int height = static_cast<int>(std::round(std::abs(y2 - y1)));

        if (width <= 0 || height <= 0) continue;

        YOLOv8Detector::BoundingBox bbox;
        bbox.score = score;
        bbox.index = class_id;
        bbox.rect = cv::Rect(left, top, width, height);
        bboxes.push_back(bbox);
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
extern "C" __declspec(dllexport) int __stdcall total_frame_count = 0;
// 必要に応じて初期化コードを追加
extern "C" __declspec(dllexport) int __stdcall get_total_frame_count() {
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

// モザイク処理
// ROI領域内のピクセルを縮小して再拡大することでピクセル化を実現
void applyMosaic(cv::Mat& frame, const cv::Rect& roiRect, int mosaicFactor = 10) {
    // roiRect の範囲が画像サイズ内か確認するのが望ましい
    cv::Mat roi = frame(roiRect);

    // 新しいサイズを計算
    int newWidth = roi.cols / mosaicFactor;
    int newHeight = roi.rows / mosaicFactor;
    if (newWidth <= 0 || newHeight <= 0) {
        // サイズが小さすぎる場合は処理しないか、強制的に1ピクセル以上とする
        //std::cerr << "applyMosaic: ROI too small (" << roi.cols << "x" << roi.rows << ")" << std::endl;
        return; // 処理スキップ
    }

    cv::Mat frame_small;
    // 小さいサイズにリサイズ（縮小）
    cv::resize(roi, frame_small, cv::Size(roi.cols / mosaicFactor, roi.rows / mosaicFactor), 0, 0, cv::INTER_NEAREST);
    // 元の大きさに補完なしで拡大（ピクセル化を強調）
    cv::resize(frame_small, roi, cv::Size(roi.cols, roi.rows), 0, 0, cv::INTER_NEAREST);
}

// ブラー処理（ガウシアンブラー）
// ROI の範囲内に対して、指定のカーネルサイズでぼかし処理を行う
void applyBlur(cv::Mat& frame, const cv::Rect& roiRect, int kernelSize = 15) {

    cv::Mat roi = frame(roiRect);

    // カーネルサイズが ROI サイズより大きくなりすぎないように制限（例: ROI 幅、高さの半分まで）
    int maxKernelWidth = roi.cols | 1;   // 奇数にするため補正が必要なら (roi.cols > 0 ? roi.cols - (roi.cols % 2 == 0 ? 1:0) : 1) とする
    int maxKernelHeight = roi.rows | 1;
    if (kernelSize > maxKernelWidth || kernelSize > maxKernelHeight) {
        kernelSize = std::min(maxKernelWidth, maxKernelHeight);
        if (kernelSize % 2 == 0) {
            kernelSize--;
        }
        if (kernelSize < 1) {
            //std::cerr << "applyBlur: ROI too small for blurring" << std::endl;
            return;
        }
    }

    // カーネルサイズは奇数でなければならない
    if (kernelSize % 2 == 0) {
        kernelSize++;
    }

    cv::GaussianBlur(roi, roi, cv::Size(kernelSize, kernelSize), 0);
}



// preview_api関数を宣言
extern "C" __declspec(dllexport) MY_API int __stdcall preview_api(const char* image_path, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool copyright, char* blacked_type, char* fixframe_type, int blacked_param, int fixframe_param)
{
	std::string imagePathStr(image_path);
    // モデルファイルのパス
    const char* model_path = "./my_yolov8m_s.onnx";

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
    //cv::Mat o_image = image.clone();
    cv::Mat o_image;

    //if (no_inference==true)
    if (strcmp(blacked_type, "No_Inference") != 0)
	{
		// 画像の前処理
		//detector.PreProcess(image, 1280, o_image);
        detector.PreProcess2(image, 1280, 736, o_image);
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

            float scoreThreshold = 0.15f; // スコアのしきい値（適宜変更）
            //float nmsThreshold = 0.5f;   // NMS のしきい値（適宜変更）
            // yolo26 は出力側で NMS が不要なので、ここで cv::dnn::NMSBoxes を呼ばない。
            // cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);

            // 代わりに全検出を利用する（必要なら閾値で絞る）
            indices.resize(scores.size());
            std::iota(indices.begin(), indices.end(), 0);
			//if (inpaint == true)
            if (strcmp(blacked_type, "Inpaint") == 0)
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
            else if (strcmp(blacked_type, "Mosaic") == 0)
            {
                 // 画像全体の矩形を生成
                cv::Rect shrinkImgRect(1, 1, image.cols - 2, image.rows - 2);

                // モザイク処理を適用
                for (const auto& i : indices) {
                    cv::Rect rect = boxes[i];
                    // ROIと画像全体の交差部分を取得
                    cv::Rect validRect = boxes[i] & shrinkImgRect;

                    // validRectのサイズが所定の閾値以上かどうかチェックする
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整

                        // 画像サイズなどに十分余裕があるか確認したほうがよい
                        applyMosaic(image, validRect, DEFAULT_MOSAIC_FACTOR / blacked_param); // mosaicFactor は適宜調整
                    }
                }
            }
            else if (strcmp(blacked_type, "Blur") == 0)
            {
                // 画像全体の矩形を生成
                cv::Rect shrinkImgRect(1, 1, image.cols - 2, image.rows - 2);

                // ブラー処理を適用
                for (const auto& i : indices) {
                    cv::Rect rect = boxes[i];
                    // ROIと画像全体の交差部分を取得
                    cv::Rect validRect = boxes[i] & shrinkImgRect;

                    // validRectのサイズが所定の閾値以上かどうかチェックする
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整

                        applyBlur(image, validRect, DEFAULT_BLUR_FACTOR * blacked_param); // カーネルサイズは適宜調整
                    }
                }
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
        if (strcmp(fixframe_type, "Mosaic") == 0) {
            // rectのサイズが所定の閾値以上かどうかチェックする
            if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整
                // モザイク処理を適用
                applyMosaic(image, rect, DEFAULT_MOSAIC_FACTOR / fixframe_param); // fixframe_param -> mosaicFactor
            }
        }
        else if (strcmp(fixframe_type, "Blur") == 0) {
            // rectのサイズが所定の閾値以上かどうかチェックする
            if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整
                // ブラー処理を適用
                applyBlur(image, rect, DEFAULT_BLUR_FACTOR * fixframe_param); // fixframe_param -> kernelSize
            }
        }
        else {
            // 単色で塗りつぶし
            cv::rectangle(image, rect, fixframe_Scalar, -1);
        }
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


// ffmpeg の標準出力（子プロセスの出力）をパイプでリダイレクトしてプロセスを起動する
HANDLE StartProcessWithRedirectedStdout(const std::string& commandLine, PROCESS_INFORMATION& pi) {
    SECURITY_ATTRIBUTES saAttr = { sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE hChildStdOutRead = NULL;
    HANDLE hChildStdOutWrite = NULL;
    if (!CreatePipe(&hChildStdOutRead, &hChildStdOutWrite, &saAttr, 0)) {
        //std::cerr << "CreatePipe (stdout) failed." << std::endl;
        return NULL;
    }
    // 親側のハンドルが子に継承されないようにする
    if (!SetHandleInformation(hChildStdOutRead, HANDLE_FLAG_INHERIT, 0)) {
        //std::cerr << "SetHandleInformation (stdout) failed." << std::endl;
        return NULL;
    }
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    // 子プロセスの標準出力（とエラー出力）を pipe の書き側に割り当てる
    si.hStdOutput = hChildStdOutWrite;
    si.hStdError = hChildStdOutWrite;
    si.dwFlags |= STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;  // 非表示に設定
    ZeroMemory(&pi, sizeof(pi));

    // ここで、std::string を変更可能なバッファ（vector）に変換する
    std::vector<char> cmdBuffer(commandLine.begin(), commandLine.end());
    cmdBuffer.push_back('\0');  // ヌル文字を追加して終端を保証

    if (!CreateProcessA(
        NULL,
        cmdBuffer.data(), // 変更可能なバッファを渡す
        NULL,
        NULL,
        TRUE,   // 子プロセスにもハンドルを継承させる
        CREATE_NO_WINDOW,  // コンソールウィンドウを表示させない
        NULL,
        NULL,
        &si,
        &pi))
    {
        //std::cerr << "CreateProcess (stdout redirect) failed with error: " << GetLastError() << std::endl;
        CloseHandle(hChildStdOutWrite);
        CloseHandle(hChildStdOutRead);
        return NULL;
    }
    // 親側では書き側は不要なのでクローズする
    CloseHandle(hChildStdOutWrite);
    return hChildStdOutRead;
}

// ffmpeg の標準入力（子プロセスが受け取る入力）をパイプでリダイレクトしてプロセスを起動する
HANDLE StartProcessWithRedirectedStdin(const std::string& commandLine, PROCESS_INFORMATION& pi) {
    SECURITY_ATTRIBUTES saAttr = { sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE hChildStdInRead = NULL;
    HANDLE hChildStdInWrite = NULL;
    if (!CreatePipe(&hChildStdInRead, &hChildStdInWrite, &saAttr, 0)) {
        //std::cerr << "CreatePipe (stdin) failed." << std::endl;
        return NULL;
    }
    // 子プロセスに継承される書き側は不要なので、親側では SetHandleInformation で外す
    if (!SetHandleInformation(hChildStdInWrite, HANDLE_FLAG_INHERIT, 0)) {
        //std::cerr << "SetHandleInformation (stdin) failed." << std::endl;
        return NULL;
    }
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    // 子プロセスはパイプの読み側から標準入力を受け取る
    si.hStdInput = hChildStdInRead;
    si.dwFlags |= STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;  // 非表示に設定
    ZeroMemory(&pi, sizeof(pi));

    // ここで、std::string を変更可能なバッファ（vector）に変換する
    std::vector<char> cmdBuffer(commandLine.begin(), commandLine.end());
    cmdBuffer.push_back('\0');  // ヌル文字を追加して終端を保証

    if (!CreateProcessA(
        NULL,
        cmdBuffer.data(), // 変更可能なバッファを渡す
        NULL,
        NULL,
        TRUE,  // ハンドルの継承を有効にする
        CREATE_NO_WINDOW,  // コンソールウィンドウを表示させない
        NULL,
        NULL,
        &si,
        &pi))
    {
        //std::cerr << "CreateProcess (stdin redirect) failed with error: " << GetLastError() << std::endl;
        CloseHandle(hChildStdInRead);
        CloseHandle(hChildStdInWrite);
        return NULL;
    }
    // 親側は読み側を不要とするのでクローズ
    CloseHandle(hChildStdInRead);
    return hChildStdInWrite;
}

// 子プロセス（ffmpeg）のパイプから1フレーム分の raw video を読み出す関数
cv::Mat ReadFrameFromPipe(HANDLE hPipe, int width, int height) {
    size_t frameSize = static_cast<size_t>(width * height * 3);
    std::vector<unsigned char> buffer(frameSize);
    size_t totalBytesRead = 0;
    DWORD bytesRead = 0;

    while (totalBytesRead < frameSize) {
        BOOL success = ReadFile(hPipe, buffer.data() + totalBytesRead,
            static_cast<DWORD>(frameSize - totalBytesRead),
            &bytesRead, NULL);
        if (!success) {
            DWORD err = GetLastError();
            if (err == ERROR_BROKEN_PIPE) {
                // EOF に到達したと判断
                //std::cerr << "ReadFile returned 0 with ERROR_BROKEN_PIPE." << std::endl;
                break;
            }
            else {
                //std::cerr << "ReadFile failed with error: " << err << std::endl;
                break;
            }
        }
        if (bytesRead == 0) {
            // データがもう来ないので終了
            //std::cerr << "ReadFile returned 0 bytes." << std::endl;
            break;
        }
        totalBytesRead += bytesRead;
    }

    if (totalBytesRead != frameSize) {
        //std::cerr << "ReadFile incomplete. Expected " << frameSize
            //<< " bytes, got " << totalBytesRead << std::endl;
        // ※デバッグ用に受け取った内容を出力可能
        return cv::Mat();  // 空のフレームを返してループ抜けにする
    }

    cv::Mat frame(height, width, CV_8UC3);
    std::memcpy(frame.data, buffer.data(), frameSize);
    return frame;
}



// 子プロセス（ffmpeg）のパイプへ1フレーム分の raw video を書き込む関数

constexpr size_t PIPE_WRITE_CHUNK = 8 * 1024 * 1024; // 8MB

void WriteFrameToPipe(HANDLE hPipe, const cv::Mat& frame) {
    if (frame.empty() || frame.data == nullptr) {
        //std::cerr << "WriteFrameToPipe: frame is empty or data is null." << std::endl;
        return;
    }
    size_t dataSize = frame.total() * frame.elemSize();
    const unsigned char* dataPtr = frame.data;
    size_t totalWritten = 0;
    DWORD bytesWritten = 0;

    while (totalWritten < dataSize) {
        DWORD toWrite = static_cast<DWORD>(std::min<size_t>(dataSize - totalWritten, PIPE_WRITE_CHUNK));
        BOOL success = WriteFile(hPipe, dataPtr + totalWritten, toWrite, &bytesWritten, NULL);
        if (!success || bytesWritten == 0) {
            //std::cerr << "WriteFile failed or incomplete write. Expected " << dataSize
                //<< " bytes, wrote " << totalWritten + bytesWritten
                //<< " (error=" << GetLastError() << ")" << std::endl;
            break;
        }
        totalWritten += bytesWritten;
    }
    if (totalWritten != dataSize) {
        //std::cerr << "WriteFrameToPipe: incomplete write. Expected " << dataSize
            //<< " bytes, wrote " << totalWritten << std::endl;
    }
}

inline void SafeCloseHandle(HANDLE& h)
{
    if (h && h != INVALID_HANDLE_VALUE ) {
        CloseHandle(h);
        h = nullptr;
    }
}

// _popen を使わずに、ffmpeg プロセスを起動しリダイレクトされたパイプ (ハンドル) を渡すユーティリティ関数群
void run_ffmpeg_input(const std::string& cmd, std::function<void(HANDLE)> func) {
    PROCESS_INFORMATION pi;
    HANDLE hPipe = StartProcessWithRedirectedStdout(cmd, pi);
    if (hPipe == NULL) {
        //std::cerr << "Failed to start ffmpeg input process." << std::endl;
        return;
    }

    // 取得したプロセスハンドルをグローバル変数に保存
    g_ffmpegInputProcessHandle = pi.hProcess;

    // ここで hPipe を使って、ffmpeg が出力する raw video を読み出す
    func(hPipe);
    // プロセス終了を待機
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
        //std::cerr << "ffmpeg input process exited with code: " << exitCode << std::endl;
    }

	if (cancelFlag != true)
	{
		// 既にキャンセル側で閉じられている可能性もあるので、安全に閉じる
		SafeCloseHandle(pi.hProcess);
		SafeCloseHandle(pi.hThread);
	}
}

void run_ffmpeg_output(const std::string& cmd, std::function<void(HANDLE)> func) {
    PROCESS_INFORMATION pi;
    HANDLE hPipe = StartProcessWithRedirectedStdin(cmd, pi);
    if (hPipe == NULL) {
        //std::cerr << "Failed to start ffmpeg output process." << std::endl;
        return;
    }

    // 取得したプロセスハンドルをグローバル変数に保存
    g_ffmpegOutputProcessHandle = pi.hProcess;

    // 書き込み用パイプ hPipe を使って、raw video を ffmpeg に書き込む
    func(hPipe);
    //// パイプを閉じることで ffmpeg 側に EOF を通知する
    //CloseHandle(hPipe);
    WaitForSingleObject(pi.hProcess, INFINITE);

    if (cancelFlag != true)
    {
        // 既にキャンセル側で閉じられている可能性もあるので、安全に閉じる
        SafeCloseHandle(pi.hProcess);
        SafeCloseHandle(pi.hThread);
    }
}

// DLL から公開する関数（エクスポート指定をするなど）
extern "C" __declspec(dllexport)
bool __stdcall CancelFfmpegProcesses(void) {
    bool result = true;
    cancelFlag = true;
    // 入力用プロセスが存在していれば終了
    if (g_ffmpegInputProcessHandle != nullptr) {
        if (!TerminateProcess(g_ffmpegInputProcessHandle, 0)) {
            //std::cerr << "Failed to terminate ffmpeg input process." << std::endl;
            result = false;
        }
        else {
            CloseHandle(g_ffmpegInputProcessHandle);
            //SafeCloseHandle(g_ffmpegInputProcessHandle);
            g_ffmpegInputProcessHandle = nullptr;
        }
    }

    // 出力用プロセスが存在していれば終了
    if (g_ffmpegOutputProcessHandle != nullptr) {
        if (!TerminateProcess(g_ffmpegOutputProcessHandle, 0)) {
            //std::cerr << "Failed to terminate ffmpeg output process." << std::endl;
            result = false;
        }
        else {
            CloseHandle(g_ffmpegOutputProcessHandle);
			//SafeCloseHandle(g_ffmpegOutputProcessHandle);
            g_ffmpegOutputProcessHandle = nullptr;
        }
    }
    //cancelFlag = true;
    return result;
}

void dml_process_frame(const cv::Mat& in_frame, cv::Mat& out_frame, YOLOv8Detector& detector, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool copyright, char* blacked_type, char* fixframe_type, int blacked_param, int fixframe_param)
{
    // OpenCV の色を作成
    cv::Scalar fixframe_Scalar(fixframe_color.b, fixframe_color.g, fixframe_color.r);
    out_frame = in_frame.clone();

    //std::cerr << "dml_process_frame: Start processing frame" << std::endl;

    //if (no_inference == true)
    if (strcmp(blacked_type , "No_Inference") != 0)
    {
        cv::Scalar name_color_Scalar(name_color.b, name_color.g, name_color.r);

        cv::Mat processed_frame;

        detector.PreProcess2(const_cast<cv::Mat&>(in_frame), 1280, 736, processed_frame);

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

            float scoreThreshold = 0.15f; // スコアのしきい値（適宜変更）
            //float nmsThreshold = 0.5f;   // NMS のしきい値（適宜変更）
            //cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
            indices.resize(scores.size());
            std::iota(indices.begin(), indices.end(), 0);

            if (strcmp(blacked_type, "Inpaint") == 0)
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
            else if (strcmp(blacked_type, "Mosaic") == 0)
            {
                // 画像全体の矩形を生成
                cv::Rect imageRect(0, 0, in_frame.cols, in_frame.rows);

                // モザイク処理を適用
                for (const auto& i : indices) {
                    cv::Rect rect = boxes[i];
                    // ROIと画像全体の交差部分を取得
                    cv::Rect validRect = boxes[i] & imageRect;

                    // validRectのサイズが所定の閾値以上かどうかチェックする
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整

                        // 画像サイズなどに十分余裕があるか確認したほうがよい
                        applyMosaic(out_frame, validRect, DEFAULT_MOSAIC_FACTOR / blacked_param); // mosaicFactor は適宜調整
                    }
                }
            }
            else if (strcmp(blacked_type, "Blur") == 0)
            {
                // 画像全体の矩形を生成
                cv::Rect imageRect(0, 0, in_frame.cols, in_frame.rows);

                // ブラー処理を適用
                for (const auto& i : indices) {
                    cv::Rect rect = boxes[i];
                    // ROIと画像全体の交差部分を取得
                    cv::Rect validRect = boxes[i] & imageRect;

                    // validRectのサイズが所定の閾値以上かどうかチェックする
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整

                        applyBlur(out_frame, validRect, DEFAULT_BLUR_FACTOR * blacked_param); // カーネルサイズは適宜調整
                    }
                }
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

        if (strcmp(fixframe_type, "Mosaic") == 0) {
            // rectのサイズが所定の閾値以上かどうかチェックする
            if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整
                // モザイク処理を適用
                applyMosaic(out_frame, rect, DEFAULT_MOSAIC_FACTOR / fixframe_param); // fixframe_param -> mosaicFactor
            }
        }
        else if (strcmp(fixframe_type, "Blur") == 0) {
            // rectのサイズが所定の閾値以上かどうかチェックする
            if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE) {  // 必要に応じて閾値を調整
                // ブラー処理を適用
                applyBlur(out_frame, rect, DEFAULT_BLUR_FACTOR * fixframe_param); // fixframe_param -> kernelSize
            }
        }
        else {
            // 単色で塗りつぶし
            cv::rectangle(out_frame, rect, fixframe_Scalar, -1);
        }

    }

    if (copyright == true)
    {
        // 貼り付ける画像を読み込む
        //cv::Mat c_sqex_image = cv::imread("C_SQUARE_ENIX.png");
        if (!c_sqex_image.empty()) {
            // 画像の幅と高さ
            int w = out_frame.cols;
            int h = out_frame.rows;
            // 関数を呼び出して画像を貼り付け
            out_frame = put_C_SQUARE_ENIX(out_frame, c_sqex_image, w, h);
        }
    }

    //std::cerr << "dml_process_frame: Finished processing frame" << std::endl;
}

// ワイド文字列 (std::wstring) を UTF-8 の std::string に変換する関数
std::string WideStringToString(const std::wstring& wstr) {
    if (wstr.empty())
        return std::string();

    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()),
        nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()),
        &strTo[0], size_needed, nullptr, nullptr);
    return strTo;
}

// 名前を変更して、DLL のパスを取得する関数
std::string GetMyDllDirectory() {
    wchar_t buffer[MAX_PATH] = { 0 };
    HMODULE hModule = nullptr;

    if (GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetMyDllDirectory),
        &hModule))
    {
        GetModuleFileNameW(hModule, buffer, MAX_PATH);
        std::wstring wFullPath(buffer);
        std::string fullPath = WideStringToString(wFullPath);
        size_t pos = fullPath.find_last_of("\\/");
        if (pos != std::string::npos) {
            return fullPath.substr(0, pos);
        }
    }
    return "";
}

//DirectMLを使用した物体検出処理
extern "C" __declspec(dllexport) MY_API int __stdcall dml_main(char* input_video_path, char* output_video_path, char* codec, char* hwaccel, int width, int height, int fps, char* color_primaries, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool copyright, char* blacked_type, char* fixframe_type, int blacked_param, int fixframe_param, char* bitrate, char* preset)
{
    const char* model_path = "my_yolov8m_s.onnx";

    YOLOv8Detector detector;


    if (!detector.loadModel(model_path)) {
        //std::cerr << "Failed to load model." << std::endl;
        return -1;
    }
    // DLLのディレクトリを取得
    std::string dllPath = GetMyDllDirectory();
    // ffmpeg.exe のフルパスを構築
    std::string ffmpegPath = dllPath + "\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe";
    // パス全体をダブルクォーテーションで囲む
    std::string quotedFfmpegExePath = "\"" + ffmpegPath + "\"";
    
    std::string ffmpeg_input_cmd;
    if (std::string(color_primaries) =="bt2020") {
        ffmpeg_input_cmd = std::string(quotedFfmpegExePath) +" -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-vf \"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }
    else {
        ffmpeg_input_cmd = std::string(quotedFfmpegExePath) + " -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }

    std::string ffmpeg_output_cmd = std::string(quotedFfmpegExePath) + " -loglevel quiet -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) +
         " -r " + std::to_string(fps) + " -i pipe:0 -pix_fmt yuv420p -vcodec " + std::string(codec) +
        " -b:v " + std::string(bitrate) + " -preset " + std::string(preset) + " \"" + std::string(output_video_path) + "\"";

    cv::Mat current_frame;
    cv::Mat processed_frame;
    total_frame_count = 0;
    cancelFlag = false;

    // フレームを保存するキュー
    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool finished_reading = false;
    const int max_queue_size = 100; // 適宜設定

     // 読み取りスレッド：ffmpeg_input の出力（raw video）を読み込む
    std::thread read_thread([&]() {
        run_ffmpeg_input(ffmpeg_input_cmd, [&](HANDLE input_pipe) {
            while (!cancelFlag) {
                cv::Mat frame = ReadFrameFromPipe(input_pipe, width, height);
                if (frame.empty()) {
                    break;
                }
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    //std::cerr << "Write thread loop check: frame_queue.size() = " << frame_queue.size()
                        //<< ", finished_reading = " << finished_reading << std::endl;
                    queue_cv.wait(lock, [&]() { return frame_queue.size() < max_queue_size; });
                    frame_queue.push(frame);
                }
                queue_cv.notify_one();
            }
            // ここでパイプを閉じることで、ffmpeg に EOF を通知する
            //std::cerr << "Read thread: closing input pipe handle to signal EOF to ffmpeg." << std::endl;
            CloseHandle(input_pipe);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                finished_reading = true;
            }
            queue_cv.notify_all();
            });
        });

    // 少し待機する（例：200ミリ秒の遅延）
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));


    // 書き込みスレッド：ffmpeg_output の入力側へ処理済みフレームを書き込む
    std::atomic<bool> writing_frame = false; // 書き込み中フラグを追加

    std::thread write_thread([&]() {
        run_ffmpeg_output(ffmpeg_output_cmd, [&](HANDLE output_pipe) {
            while (!cancelFlag) {
                cv::Mat processed_frame;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    // 1秒間待っても返事がなければタイムアウトとする
                    if (!queue_cv.wait_for(lock, std::chrono::milliseconds(2000), [&]() {
                        return !frame_queue.empty() || finished_reading;
                        })) {
                        // タイムアウト時、書き込み中なら終了しない
                        if (writing_frame) {
                            //std::cerr << "Write thread: wait_for timed out, but writing in progress. Continue waiting." << std::endl;
                            continue;
                        } else {
                            //std::cerr << "Write thread: wait_for timed out (no data for 3 seconds, not writing). Exiting." << std::endl;
                            break;
                        }
                    }
                    if (frame_queue.empty() && finished_reading) {
                        //std::cerr << "Exit write loop: frame queue empty and finished_reading is true." << std::endl;
                        break;
                    }
                    if (!frame_queue.empty()) {
                        processed_frame = frame_queue.front();
                        frame_queue.pop();
                    }
                }

                if (!processed_frame.empty()) {
                    writing_frame = true; // 書き込み開始
                    try {
                        dml_process_frame(processed_frame, processed_frame, detector, rects, count, name_color, fixframe_color, copyright, blacked_type, fixframe_type, blacked_param, fixframe_param);
                        total_frame_count += 1;
                        WriteFrameToPipe(output_pipe, processed_frame);
                    } catch (const std::exception& e) {
                        //std::cerr << "Exception in write thread: " << e.what() << std::endl;
                    }
                    writing_frame = false; // 書き込み終了
                }
                queue_cv.notify_one();
            }
            queue_cv.notify_one();
            //std::cerr << "Write thread: closing output pipe handle to signal EOF to ffmpeg." << std::endl;
            CloseHandle(output_pipe);
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
    const cv::Size& newShape = cv::Size(1280, 736),
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
        //outImage = image.clone();
        outImage = image;
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


void postprocess(float* rst, int batch_size, std::vector<cv::Mat>& images, std::vector<cv::Vec4d>& params, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool copyright, char* blacked_type, char* fixframe_type, int blacked_param, int fixframe_param, int out_d1, int out_d2)
{
    cv::Scalar fixframe_Scalar(fixframe_color.b, fixframe_color.g, fixframe_color.r);

    // out_d1 = detections (例: 300), out_d2 = per_det (例: 6)
    if (out_d2 < 6) return; // 想定外のフォーマットなら即時 return

    const float score_threshold = 0.15f;
    //const float nms_threshold = 0.5f;

    for (int b = 0; b < batch_size; ++b) {
        // バッチごとに出力ベースを計算
        const float* base = rst + static_cast<size_t>(b) * static_cast<size_t>(out_d1) * static_cast<size_t>(out_d2);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        if (strcmp(blacked_type, "No_Inference") != 0) {
            for (int i = 0; i < out_d1; ++i) {
                const float* det = base + static_cast<size_t>(i) * static_cast<size_t>(out_d2);
                float score = det[4];
                if (score < score_threshold) continue;

                float x1_model = det[0];
                float y1_model = det[1];
                float x2_model = det[2];
                float y2_model = det[3];
                int class_id = static_cast<int>(det[5]);

                // params[b] は LetterBox の戻り値:
                // params[b] = [ratio_x, ratio_y, pad_w(left), pad_h(top)]
                float ratio_x = static_cast<float>(params[b][0]);
                float ratio_y = static_cast<float>(params[b][1]);
                float pad_w = static_cast<float>(params[b][2]);
                float pad_h = static_cast<float>(params[b][3]);

                // モデル座標（レターボックスされた入力画像上のピクセル座標） -> 元画像座標
                float x1 = (x1_model - pad_w) / ratio_x;
                float y1 = (y1_model - pad_h) / ratio_y;
                float x2 = (x2_model - pad_w) / ratio_x;
                float y2 = (y2_model - pad_h) / ratio_y;

                int left = std::max(0, static_cast<int>(std::round(std::min(x1, x2))));
                int top = std::max(0, static_cast<int>(std::round(std::min(y1, y2))));
                int width = static_cast<int>(std::round(std::abs(x2 - x1)));
                int height = static_cast<int>(std::round(std::abs(y2 - y1)));

                if (width <= 0 || height <= 0) continue;

                boxes.emplace_back(left, top, width, height);
                scores.push_back(score);
                class_ids.push_back(class_id);
            }

            // NMS
            std::vector<int> indices;
            //cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
            // yolo26 は NMS 不要のため、全インデックスを使う
            indices.resize(scores.size());
            std::iota(indices.begin(), indices.end(), 0);

            // apply operations based on blacked_type
            if (strcmp(blacked_type, "Inpaint") == 0) {
                cv::Mat mask_frame = cv::Mat::zeros(images[b].size(), CV_8UC1);
                for (int idx : indices) {
                    cv::rectangle(mask_frame, boxes[idx], cv::Scalar(255), cv::FILLED);
                }
                cv::Mat inpainted_image;
                cv::inpaint(images[b], mask_frame, inpainted_image, 3, cv::INPAINT_TELEA);
                images[b] = inpainted_image;
            }
            else if (strcmp(blacked_type, "Mosaic") == 0) {
                cv::Rect imageRect(0, 0, images[b].cols, images[b].rows);
                for (int idx : indices) {
                    cv::Rect validRect = boxes[idx] & imageRect;
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE)
                        applyMosaic(images[b], validRect, DEFAULT_MOSAIC_FACTOR / blacked_param);
                }
            }
            else if (strcmp(blacked_type, "Blur") == 0) {
                cv::Rect imageRect(0, 0, images[b].cols, images[b].rows);
                for (int idx : indices) {
                    cv::Rect validRect = boxes[idx] & imageRect;
                    if (validRect.width > MIN_VALID_RECT_SIZE && validRect.height > MIN_VALID_RECT_SIZE)
                        applyBlur(images[b], validRect, DEFAULT_BLUR_FACTOR * blacked_param);
                }
            }
            else {
                // 単純描画（name_color）
                cv::Scalar name_color_Scalar(name_color.b, name_color.g, name_color.r);
                for (int idx : indices) {
                    cv::rectangle(images[b], boxes[idx], name_color_Scalar, -1);
                }
            }
        }

        // 固定矩形描画 (既存処理)
        for (int i = 0; i < count; i++) {
            cv::Rect rect(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
            if (strcmp(fixframe_type, "Mosaic") == 0) {
                if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE)
                    applyMosaic(images[b], rect, DEFAULT_MOSAIC_FACTOR / fixframe_param);
            }
            else if (strcmp(fixframe_type, "Blur") == 0) {
                if (rect.width > MIN_VALID_RECT_SIZE && rect.height > MIN_VALID_RECT_SIZE)
                    applyBlur(images[b], rect, DEFAULT_BLUR_FACTOR * fixframe_param);
            }
            else {
                cv::rectangle(images[b], rect, fixframe_Scalar, -1);
            }
        }

        // 透かし(copyright)
        if (copyright == true && !c_sqex_image.empty()) {
            int w = images[b].cols;
            int h = images[b].rows;
            images[b] = put_C_SQUARE_ENIX(images[b], c_sqex_image, w, h);
        }
    }
}


// TensorRTを使用した物体検出処理
extern "C" __declspec(dllexport) MY_API int __stdcall trt_main(char* input_video_path, char* output_video_path, char* codec, char* hwaccel, int width, int height, int fps, char* color_primaries, RectInfo* rects, int count, ColorInfo name_color, ColorInfo fixframe_color, bool copyright, char* blacked_type, char* fixframe_type, int blacked_param, int fixframe_param, char* bitrate, char* preset)
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

    const int max_batch =16; // 最大バッチサイズを設定 (固定)
    const size_t max_queue_size =100; // キューの最大サイズを設定

    // ローカルフォルダのパスを取得
    auto localFolder = ApplicationData::Current().LocalFolder();
    std::wstring localAppDataPath = localFolder.Path().c_str();

    // アプリケーション専用のフォルダパスを組み立てる
    std::wstring appFolderPath = std::wstring(localAppDataPath) + std::wstring{ L"\\WoLNamesBlackedOut" };
    std::wstring engineFilePath = appFolderPath + std::wstring{ L"\\my_yolov8m_s_20260201.engine" };
    std::wstring casheFilePath = appFolderPath + std::wstring{ L"\\my_yolov8m_s_20260201.cashe" };

    std::string engineFilePathStr(engineFilePath.length(),0);
    std::transform(engineFilePath.begin(), engineFilePath.end(), engineFilePathStr.begin(), [](wchar_t c) {
    return (char)c;
    });
    const char* engineFilePathCStr = engineFilePathStr.c_str();

    std::string casheFilePathStr(casheFilePath.length(),0);
    std::transform(casheFilePath.begin(), casheFilePath.end(), casheFilePathStr.begin(), [](wchar_t c) {
    return (char)c;
    });
    const char* casheFilePathCStr = casheFilePathStr.c_str();

	//エンジンファイルの読み込み
    auto plan = load_engine_file(engineFilePathCStr);

    //キャッシュファイルの有無確認用
    std::ifstream ifs(casheFilePathStr);

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (engine == nullptr) { return false; }

    //ランタイムキャッシュ利用
    nvinfer1::IRuntimeConfig* runtimeConfig = engine->createRuntimeConfig();
    runtimeConfig->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC);
    nvinfer1::IRuntimeCache* runtimeCache = runtimeConfig->createRuntimeCache();

    if (ifs.good()) { //キャッシュがある場合、キャッシュを読み込む
        auto casheplan = load_engine_file(casheFilePathCStr);
        runtimeCache->deserialize(
            casheplan.data(), casheplan.size());
    }

    runtimeConfig->setRuntimeCache(*runtimeCache);
    runtimeConfig->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);   //TensorRT-RTX1.20以降
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext(runtimeConfig));
    if (context == nullptr) { return false; }

    auto idims = engine->getTensorShape("images");
    auto odims = engine->getTensorShape("output0");
    nvinfer1::Dims4 inputDims = { max_batch, idims.d[1], idims.d[2], idims.d[3] };
    nvinfer1::Dims3 outputDims = { max_batch, odims.d[1], odims.d[2] };
    context->setInputShape("images", inputDims);

    void* buffers[2];
    const int inputIndex =0;
    const int outputIndex =1;

    // GPUバッファは max_batch 基準で確保
    cudaMalloc(&buffers[inputIndex], max_batch * idims.d[1] * idims.d[2] * idims.d[3] * sizeof(float));
    cudaMalloc(&buffers[outputIndex], max_batch * odims.d[1] * odims.d[2] * sizeof(float));

    context->setTensorAddress("images", buffers[inputIndex]);
    context->setTensorAddress("output0", buffers[outputIndex]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // DLLのディレクトリを取得
    std::string dllPath = GetMyDllDirectory();
    // ffmpeg.exe のフルパスを構築
    std::string ffmpegPath = dllPath + "\\ffmpeg-master-latest-win64-lgpl\\bin\\ffmpeg.exe";
    // パス全体をダブルクォーテーションで囲む
    std::string quotedFfmpegExePath = "\"" + ffmpegPath + "\"";

    std::string ffmpeg_input_cmd;
    if (std::string(color_primaries) == "bt2020") {
        ffmpeg_input_cmd = std::string(quotedFfmpegExePath) + " -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-vf \"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }
    else {

        ffmpeg_input_cmd = std::string(quotedFfmpegExePath) + " -loglevel quiet -hwaccel " + std::string(hwaccel) + " -i \"" + std::string(input_video_path) + "\" " +
            "-f image2pipe -vcodec rawvideo -pix_fmt bgr24 -";
    }

    std::string ffmpeg_output_cmd = std::string(quotedFfmpegExePath) + " -loglevel quiet -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) + " -i pipe:0 -pix_fmt yuv420p -vcodec " + std::string(codec) +
        " -b:v " + std::string(bitrate) + " -preset " + std::string(preset)+ " \"" + std::string(output_video_path) + "\"";

    cv::Mat current_frame;
    cv::Mat processed_frame;
    total_frame_count = 0;

    // ホスト出力バッファも max_batch 基準で確保
    float* rst = new float[max_batch *5 *19320];

    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool finished_reading = false;
    cancelFlag = false;

    // 読み取りスレッド：ffmpeg_input の出力（raw video）を読み込む
    std::thread read_thread([&]() {
        run_ffmpeg_input(ffmpeg_input_cmd, [&](HANDLE input_pipe) {
            while (!cancelFlag) {
                cv::Mat frame = ReadFrameFromPipe(input_pipe, width, height);
                if (frame.empty()) {
                    break;
                }
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    //std::cerr << "Write thread loop check: frame_queue.size() = " << frame_queue.size()
                        //<< ", finished_reading = " << finished_reading << std::endl;
                    queue_cv.wait(lock, [&]() { return frame_queue.size() < max_queue_size; });
                    frame_queue.push(frame);
                }
                queue_cv.notify_one();
            }
            //ここでパイプを閉じることで、ffmpeg に EOF を通知する
            //std::cerr << "Read thread: closing input pipe handle to signal EOF to ffmpeg." << std::endl;
            CloseHandle(input_pipe);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                finished_reading = true;
            }
            queue_cv.notify_all();
        });
    });

    // 少し待機する（例：200ミリ秒の遅延）
    std::this_thread::sleep_for(std::chrono::milliseconds(500));


    // 書き込みスレッド：ffmpeg_output の入力側へ処理済みフレームを書き込む
    std::atomic<bool> writing_frame = false; // 書き込み中フラグを追加

    std::thread write_thread([&]() {
        run_ffmpeg_output(ffmpeg_output_cmd, [&](HANDLE output_pipe) {
            while (!cancelFlag) {
                std::vector<cv::Mat> frames;
                std::vector<cv::Vec4d> params;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    //1秒間待ってもデータがなければタイムアウトとして終了する
                    if (!queue_cv.wait_for(lock, std::chrono::milliseconds(2000), [&]() {
                        return !frame_queue.empty() || finished_reading;
                    })) {
                        // タイムアウト時、書き込み中なら終了しない
                        if (writing_frame) {
                            //std::cerr << "Write thread: wait_for timed out, but writing in progress. Continue waiting." << std::endl;
                            continue;
                        }
                        else {
                            //std::cerr << "Write thread: wait_for timed out (no data for3 seconds, not writing). Exiting." << std::endl;
                            break;
                        }
                    }
                    if (frame_queue.empty() && finished_reading) {
                        //std::cerr << "Exit write loop: frame queue empty and finished_reading is true." << std::endl;
                        break;
                    }
                    while (!frame_queue.empty() && frames.size() < max_batch) {
                        frames.push_back(frame_queue.front());
                        frame_queue.pop();
                    }
                }
                if (frames.empty()) {
                    //std::cerr << "No frames to process, exiting write loop." << std::endl;
                    break;
                }
                writing_frame = true; // 書き込み開始
                std::vector<cv::Mat> blobs;
                for (auto& frame : frames) {
                    cv::Mat LetterBoxImg;
                    cv::Vec4d param;
                    LetterBox(frame, LetterBoxImg, param, cv::Size(1280,736));
                    params.push_back(param);

                    cv::Mat blob;
                    cv::dnn::blobFromImage(LetterBoxImg, blob,1 /255.0, cv::Size(1280,736), cv::Scalar(0,0,0), true, false, CV_32F);
                    blobs.push_back(blob);
                }

                int n = static_cast<int>(blobs.size()); // 実際のバッチサイズ

                // 実バッチに合わせて入力形状を設定
                nvinfer1::Dims4 dynamicInputDims = { n, idims.d[1], idims.d[2], idims.d[3] };
                context->setInputShape("images", dynamicInputDims);

                // ホスト -> デバイスへ n 件分コピー
                for (int i =0; i < n; ++i) {
                    cudaMemcpyAsync(static_cast<float*>(buffers[inputIndex]) + i * idims.d[1] * idims.d[2] * idims.d[3], blobs[i].data, idims.d[1] * idims.d[2] * idims.d[3] * sizeof(float), cudaMemcpyHostToDevice, stream);
                }

                context->setOptimizationProfileAsync(0, stream);
                context->enqueueV3(stream);
                //cudaStreamSynchronize(stream);

                // デバイス -> ホストへ n 件分だけコピー
                size_t outPerImg = odims.d[1] * odims.d[2];
                cudaMemcpyAsync(rst, buffers[outputIndex], n * outPerImg * sizeof(float), cudaMemcpyDeviceToHost, stream);
                //cudaStreamSynchronize(stream);

                // 後処理へ実バッチ数を渡す
                postprocess(rst, n, frames, params, rects, count, name_color, fixframe_color, copyright, blacked_type, fixframe_type, blacked_param, fixframe_param, odims.d[1], odims.d[2]);
                for (auto& frame : frames) {
                    WriteFrameToPipe(output_pipe, frame);
                }
                writing_frame = false; // 書き込み終了
                //total_frame_count += frames.size();
                total_frame_count += n;
                queue_cv.notify_one();
            }
            queue_cv.notify_one();
            //std::cerr << "Write thread: closing output pipe handle to signal EOF to ffmpeg." << std::endl;
            CloseHandle(output_pipe);
            });
        });


    read_thread.join();
    write_thread.join();

	if (!ifs.good()) { //キャッシュがない場合、キャッシュを保存
        // get the runtime config from the execution context
        nvinfer1::IRuntimeConfig* runtimeConfig1 = context->getRuntimeConfig();

        // get the runtime cache from the runtime config
        nvinfer1::IRuntimeCache* runtimeCache1 = runtimeConfig1->getRuntimeCache();

        // serialize the cache into a memory blob
        nvinfer1::IHostMemory* hostMemory = runtimeCache1->serialize();
        assert(hostMemory != nullptr);

        // save the serialized cache to disk
        std::ofstream engine_file(casheFilePathCStr, std::ios::binary);
        //assert(engine_file.is_open() && "Failed to open engine file");
        engine_file.write((char*)hostMemory->data(), hostMemory->size());
        engine_file.close();
    }

    delete[] rst;
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}

