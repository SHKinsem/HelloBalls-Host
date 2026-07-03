#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class GrayscaleConverterNode : public rclcpp::Node
{
public:
  GrayscaleConverterNode()
  : Node("helloballs_grayscale_converter")
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "/camera/image_raw");
    output_topic_ = declare_parameter<std::string>("output_topic", "/camera/image_mono");
    frame_id_override_ = declare_parameter<std::string>("frame_id", "");

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      output_topic_,
      rclcpp::SensorDataQoS());
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GrayscaleConverterNode::handleImage, this, std::placeholders::_1));

    last_log_time_ = get_clock()->now();
    RCLCPP_INFO(
      get_logger(),
      "grayscale converter forwarding %s -> %s",
      input_topic_.c_str(),
      output_topic_.c_str());
  }

private:
  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      convertAndPublish(*msg);
      frames_since_log_++;
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "failed to convert image: %s",
        e.what());
    }

    const auto t = get_clock()->now();
    if ((t - last_log_time_).seconds() >= 1.0) {
      const auto fps = frames_since_log_ / std::max((t - last_log_time_).seconds(), 1e-6);
      RCLCPP_INFO(get_logger(), "mono image FPS: %.1f", fps);
      frames_since_log_ = 0;
      last_log_time_ = t;
    }
  }

  void convertAndPublish(const sensor_msgs::msg::Image & msg)
  {
    if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
      throw std::runtime_error("empty image");
    }

    const auto encoding = normalizedEncoding(msg.encoding);
    const cv::Mat input = imageView(msg, encoding);
    cv::Mat mono;

    if (encoding == "mono8") {
      mono = input;
    } else if (encoding == "bgr8") {
      cv::cvtColor(input, mono_buffer_, cv::COLOR_BGR2GRAY);
      mono = mono_buffer_;
    } else if (encoding == "rgb8") {
      cv::cvtColor(input, mono_buffer_, cv::COLOR_RGB2GRAY);
      mono = mono_buffer_;
    } else if (encoding == "bgra8") {
      cv::cvtColor(input, mono_buffer_, cv::COLOR_BGRA2GRAY);
      mono = mono_buffer_;
    } else if (encoding == "rgba8") {
      cv::cvtColor(input, mono_buffer_, cv::COLOR_RGBA2GRAY);
      mono = mono_buffer_;
    } else {
      throw std::runtime_error("unsupported encoding: " + msg.encoding);
    }

    auto out = sensor_msgs::msg::Image();
    out.header = msg.header;
    if (!frame_id_override_.empty()) {
      out.header.frame_id = frame_id_override_;
    }
    out.height = msg.height;
    out.width = msg.width;
    out.encoding = "mono8";
    out.is_bigendian = msg.is_bigendian;
    out.step = msg.width;

    const auto row_bytes = static_cast<size_t>(out.width);
    out.data.resize(row_bytes * out.height);
    for (uint32_t row = 0; row < out.height; ++row) {
      std::memcpy(
        out.data.data() + row_bytes * row,
        mono.ptr(row),
        row_bytes);
    }
    image_pub_->publish(std::move(out));
  }

  cv::Mat imageView(const sensor_msgs::msg::Image & msg, const std::string & encoding) const
  {
    const int channels = channelsForEncoding(encoding);
    const auto min_step = static_cast<size_t>(msg.width) * channels;
    if (msg.step < min_step) {
      throw std::runtime_error("image step is smaller than width * channels");
    }
    const auto required_size = static_cast<size_t>(msg.step) * msg.height;
    if (msg.data.size() < required_size) {
      throw std::runtime_error("image data is smaller than step * height");
    }
    return cv::Mat(
      static_cast<int>(msg.height),
      static_cast<int>(msg.width),
      CV_8UC(channels),
      const_cast<unsigned char *>(msg.data.data()),
      msg.step);
  }

  int channelsForEncoding(const std::string & encoding) const
  {
    if (encoding == "mono8") {
      return 1;
    }
    if (encoding == "bgr8" || encoding == "rgb8") {
      return 3;
    }
    if (encoding == "bgra8" || encoding == "rgba8") {
      return 4;
    }
    throw std::runtime_error("unsupported encoding: " + encoding);
  }

  std::string normalizedEncoding(std::string encoding) const
  {
    std::transform(encoding.begin(), encoding.end(), encoding.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return encoding;
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string frame_id_override_;
  cv::Mat mono_buffer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Time last_log_time_{0, 0, RCL_ROS_TIME};
  int frames_since_log_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GrayscaleConverterNode>());
  rclcpp::shutdown();
  return 0;
}
