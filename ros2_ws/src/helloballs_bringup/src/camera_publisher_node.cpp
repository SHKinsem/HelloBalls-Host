#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

namespace
{

int fourccFromString(const std::string & fourcc)
{
  if (fourcc.size() != 4) {
    throw std::runtime_error("camera_fourcc must contain exactly four characters");
  }
  return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

void configureWithV4l2Ctl(
  const std::string & device,
  int width,
  int height,
  double fps,
  const std::string & fourcc)
{
  if (fourcc.empty()) {
    return;
  }

  const std::string fmt_cmd =
    "v4l2-ctl --device=" + device +
    " --set-fmt-video=width=" + std::to_string(width) +
    ",height=" + std::to_string(height) +
    ",pixelformat=" + fourcc + " >/dev/null 2>&1";
  std::ignore = std::system(fmt_cmd.c_str());

  if (fps > 0.0) {
    const std::string fps_cmd =
      "v4l2-ctl --device=" + device +
      " --set-parm=" + std::to_string(fps) + " >/dev/null 2>&1";
    std::ignore = std::system(fps_cmd.c_str());
  }
}

std::string decodeFourcc(double value)
{
  const auto fourcc = static_cast<int>(value);
  std::string out;
  out.reserve(4);
  for (int shift = 0; shift <= 24; shift += 8) {
    const char c = static_cast<char>((fourcc >> shift) & 0xFF);
    out.push_back(std::isprint(static_cast<unsigned char>(c)) ? c : '?');
  }
  return out;
}

}  // namespace

class CameraPublisherNode : public rclcpp::Node
{
public:
  CameraPublisherNode()
  : Node("helloballs_camera_publisher")
  {
    device_ = declare_parameter<std::string>("camera_device", "/dev/video0");
    width_ = declare_parameter<int>("camera_width", 1280);
    height_ = declare_parameter<int>("camera_height", 720);
    fps_ = declare_parameter<double>("camera_fps", 30.0);
    fourcc_ = declare_parameter<std::string>("camera_fourcc", "");
    frame_id_ = declare_parameter<std::string>("camera_frame_id", "camera_link");
    use_v4l2_ctl_ = declare_parameter<bool>("use_v4l2_ctl", false);
    buffer_size_ = declare_parameter<int>("camera_buffer_size", 1);
    camera_info_rate_hz_ = declare_parameter<double>("camera_info_rate_hz", 1.0);

    const auto image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const auto camera_info_topic =
      declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      image_topic,
      rclcpp::SensorDataQoS());
    camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      camera_info_topic,
      rclcpp::SensorDataQoS());

    openCamera();
    last_log_time_ = get_clock()->now();
    last_camera_info_time_ = get_clock()->now();

    const auto period = std::chrono::duration<double>(1.0 / std::max(fps_, 1.0));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CameraPublisherNode::publishFrame, this));
  }

private:
  void openCamera()
  {
    capture_.release();
    if (use_v4l2_ctl_) {
      configureWithV4l2Ctl(device_, width_, height_, fps_, fourcc_);
    }

    capture_.open(device_, cv::CAP_V4L2);
    if (!capture_.isOpened()) {
      throw std::runtime_error("failed to open camera " + device_);
    }

    if (!fourcc_.empty()) {
      capture_.set(cv::CAP_PROP_FOURCC, fourccFromString(fourcc_));
    }
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    capture_.set(cv::CAP_PROP_FPS, fps_);
    if (buffer_size_ > 0) {
      capture_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
    }

    actual_width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    actual_height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    actual_fps_ = capture_.get(cv::CAP_PROP_FPS);
    actual_fourcc_ = decodeFourcc(capture_.get(cv::CAP_PROP_FOURCC));

    cv::Mat frame;
    bool got_frame = false;
    for (int attempt = 1; attempt <= 60; ++attempt) {
      if (capture_.read(frame) && !frame.empty()) {
        got_frame = true;
        break;
      }
      if (attempt == 1 || attempt % 10 == 0) {
        RCLCPP_WARN(
          get_logger(),
          "waiting for first camera frame from %s (%d/60)",
          device_.c_str(),
          attempt);
      }
      std::this_thread::sleep_for(50ms);
    }
    if (!got_frame) {
      throw std::runtime_error("camera opened but did not return a frame after warmup");
    }

    RCLCPP_INFO(
      get_logger(),
      "OpenCV camera %s opened at %dx%d %s %.1ffps",
      device_.c_str(),
      actual_width_,
      actual_height_,
      actual_fourcc_.c_str(),
      actual_fps_);
    consecutive_read_failures_ = 0;
    reopen_backoff_ms_ = 500;
  }

  void publishFrame()
  {
    const auto read_started_at = std::chrono::steady_clock::now();
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
      consecutive_read_failures_++;
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "failed to read camera frame (%d consecutive failures)",
        consecutive_read_failures_);
      if (consecutive_read_failures_ >= 90 && consecutive_read_failures_ % 30 == 0) {
        RCLCPP_WARN(
          get_logger(),
          "reopening camera after repeated read failures; backoff=%dms",
          reopen_backoff_ms_);
        std::this_thread::sleep_for(std::chrono::milliseconds(reopen_backoff_ms_));
        try {
          openCamera();
        } catch (const std::exception & error) {
          RCLCPP_ERROR(get_logger(), "camera reopen failed: %s", error.what());
          reopen_backoff_ms_ = std::min(reopen_backoff_ms_ * 2, 5000);
        }
      }
      return;
    }
    consecutive_read_failures_ = 0;
    reopen_backoff_ms_ = 500;
    const auto read_finished_at = std::chrono::steady_clock::now();

    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp = now();
    msg.header.frame_id = frame_id_;
    msg.height = static_cast<uint32_t>(frame.rows);
    msg.width = static_cast<uint32_t>(frame.cols);
    msg.encoding = frame.channels() == 1 ? "mono8" : "bgr8";
    msg.is_bigendian = false;
    msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(frame.cols * frame.elemSize());
    const auto data_size = static_cast<size_t>(msg.step) * frame.rows;
    msg.data.resize(data_size);
    std::memcpy(msg.data.data(), frame.data, data_size);
    image_pub_->publish(std::move(msg));
    const auto publish_finished_at = std::chrono::steady_clock::now();

    read_time_sum_ += std::chrono::duration<double>(read_finished_at - read_started_at).count();
    publish_time_sum_ += std::chrono::duration<double>(publish_finished_at - read_finished_at).count();

    maybePublishCameraInfo();
    frames_since_log_++;
    const auto t = get_clock()->now();
    if ((t - last_log_time_).seconds() >= 1.0) {
      const auto fps = frames_since_log_ / (t - last_log_time_).seconds();
      const auto denom = std::max(frames_since_log_, 1);
      RCLCPP_INFO(
        get_logger(),
        "camera publish FPS: %.1f, avg read %.1fms, publish %.1fms",
        fps,
        1000.0 * read_time_sum_ / denom,
        1000.0 * publish_time_sum_ / denom);
      frames_since_log_ = 0;
      read_time_sum_ = 0.0;
      publish_time_sum_ = 0.0;
      last_log_time_ = t;
    }
  }

  void maybePublishCameraInfo()
  {
    if (camera_info_rate_hz_ <= 0.0) {
      return;
    }
    const auto t = get_clock()->now();
    if ((t - last_camera_info_time_).seconds() < 1.0 / camera_info_rate_hz_) {
      return;
    }

    auto info = sensor_msgs::msg::CameraInfo();
    info.header.stamp = t;
    info.header.frame_id = frame_id_;
    info.height = actual_height_;
    info.width = actual_width_;
    info.distortion_model = "plumb_bob";
    info.k[8] = 1.0;
    info.r[0] = 1.0;
    info.r[4] = 1.0;
    info.r[8] = 1.0;
    info.p[10] = 1.0;
    camera_info_pub_->publish(std::move(info));
    last_camera_info_time_ = t;
  }

  std::string device_;
  std::string fourcc_;
  std::string frame_id_;
  int width_{1280};
  int height_{720};
  int buffer_size_{1};
  int actual_width_{0};
  int actual_height_{0};
  double fps_{30.0};
  double actual_fps_{0.0};
  double camera_info_rate_hz_{1.0};
  bool use_v4l2_ctl_{false};
  std::string actual_fourcc_;
  cv::VideoCapture capture_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_log_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_camera_info_time_{0, 0, RCL_ROS_TIME};
  int frames_since_log_{0};
  int consecutive_read_failures_{0};
  int reopen_backoff_ms_{500};
  double read_time_sum_{0.0};
  double publish_time_sum_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CameraPublisherNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("helloballs_camera_publisher"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
