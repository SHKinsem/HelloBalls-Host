#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct Point2
{
  double x{0.0};
  double y{0.0};
};

struct Line2
{
  Point2 a;
  Point2 b;
};

struct CandidatePose
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double score{std::numeric_limits<double>::infinity()};
  std::string label;
};

double normalizeAngle(double value)
{
  while (value > kPi) {
    value -= 2.0 * kPi;
  }
  while (value < -kPi) {
    value += 2.0 * kPi;
  }
  return value;
}

double pointDistance(const Point2 & a, const Point2 & b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

Point2 transformPoint(const Point2 & p, const CandidatePose & pose)
{
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  return {pose.x + c * p.x - s * p.y, pose.y + s * p.x + c * p.y};
}

double distancePointToSegment(const Point2 & p, const Line2 & line)
{
  const double vx = line.b.x - line.a.x;
  const double vy = line.b.y - line.a.y;
  const double wx = p.x - line.a.x;
  const double wy = p.y - line.a.y;
  const double len2 = vx * vx + vy * vy;
  if (len2 <= 1e-9) {
    return pointDistance(p, line.a);
  }
  const double t = std::clamp((wx * vx + wy * vy) / len2, 0.0, 1.0);
  const Point2 projection{line.a.x + t * vx, line.a.y + t * vy};
  return pointDistance(p, projection);
}

double yawFromQuaternion(double x, double y, double z, double w)
{
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

geometry_msgs::msg::Point makePoint(double x, double y, double z)
{
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

}  // namespace

class CourtLineLocalizerNode : public rclcpp::Node
{
public:
  CourtLineLocalizerNode()
  : Node("court_line_localizer_node")
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_mono");
    camera_info_topic_ = declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    vio_odom_topic_ = declare_parameter<std::string>("vio_odom_topic", "/vio/odometry");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    court_frame_ = declare_parameter<std::string>("court_frame", "court");
    start_side_ = declare_parameter<std::string>("start_side", "unknown");
    camera_height_m_ = declare_parameter<double>("camera_height_m", 0.14214);
    camera_pitch_rad_ = declare_parameter<double>("camera_pitch_rad", 0.0);
    court_length_m_ = declare_parameter<double>("court_length_m", 23.77);
    court_width_m_ = declare_parameter<double>("court_width_m", 10.97);
    singles_width_m_ = declare_parameter<double>("singles_width_m", 8.23);
    service_line_distance_from_net_m_ =
      declare_parameter<double>("service_line_distance_from_net_m", 6.40);
    roi_start_fraction_ = declare_parameter<double>("roi_start_fraction", 0.45);
    min_hough_line_length_px_ = declare_parameter<int>("min_hough_line_length_px", 45);
    max_hough_line_gap_px_ = declare_parameter<int>("max_hough_line_gap_px", 12);
    hough_threshold_ = declare_parameter<int>("hough_threshold", 40);
    adaptive_block_size_ = declare_parameter<int>("adaptive_block_size", 31);
    adaptive_c_ = declare_parameter<double>("adaptive_c", -8.0);
    max_detected_lines_ = declare_parameter<int>("max_detected_lines", 40);
    min_projected_line_length_m_ = declare_parameter<double>("min_projected_line_length_m", 0.20);
    match_max_average_error_m_ = declare_parameter<double>("match_max_average_error_m", 0.65);
    search_xy_range_m_ = declare_parameter<double>("search_xy_range_m", 2.0);
    search_xy_step_m_ = declare_parameter<double>("search_xy_step_m", 0.25);
    search_yaw_range_rad_ = declare_parameter<double>("search_yaw_range_rad", 0.70);
    search_yaw_step_rad_ = declare_parameter<double>("search_yaw_step_rad", 0.0872664626);
    initial_side_min_v_fraction_ =
      declare_parameter<double>("initial_side_min_v_fraction", 0.65);
    initial_side_max_angle_rad_ =
      declare_parameter<double>("initial_side_max_angle_rad", 0.2617993878);
    initial_side_min_length_px_ =
      declare_parameter<double>("initial_side_min_length_px", 120.0);
    initial_side_min_confidence_ =
      declare_parameter<double>("initial_side_min_confidence", 0.35);
    initial_side_required_frames_ =
      declare_parameter<int>("initial_side_required_frames", 3);
    match_doubles_sidelines_ =
      declare_parameter<bool>("match_doubles_sidelines", true);
    full_map_min_vio_translation_m_ =
      declare_parameter<double>("full_map_min_vio_translation_m", 0.5);
    full_map_min_vio_rotation_rad_ =
      declare_parameter<double>("full_map_min_vio_rotation_rad", 0.35);

    validateParameters();
    rebuildCourtMap();
    resetCandidates();

    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>("/court/debug_image", 10);
    map_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/court/map_lines", 1);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/court/pose_measurement",
      10);

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&CourtLineLocalizerNode::handleCameraInfo, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&CourtLineLocalizerNode::handleImage, this, std::placeholders::_1));
    vio_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      vio_odom_topic_,
      50,
      std::bind(&CourtLineLocalizerNode::handleVioOdometry, this, std::placeholders::_1));

    map_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&CourtLineLocalizerNode::publishCourtMap, this));

    RCLCPP_INFO(
      get_logger(),
      "court line localizer listening image=%s camera_info=%s vio=%s start_side=%s",
      image_topic_.c_str(),
      camera_info_topic_.c_str(),
      vio_odom_topic_.c_str(),
      start_side_.c_str());
  }

private:
  void validateParameters()
  {
    if (
      start_side_ != "unknown" && start_side_ != "sideline_left" &&
      start_side_ != "sideline_right")
    {
      throw std::runtime_error(
              "start_side must be unknown, sideline_left, or sideline_right");
    }
    adaptive_block_size_ = std::max(3, adaptive_block_size_ | 1);
    roi_start_fraction_ = std::clamp(roi_start_fraction_, 0.0, 0.9);
    max_detected_lines_ = std::max(1, max_detected_lines_);
    search_xy_step_m_ = std::max(0.05, search_xy_step_m_);
    search_yaw_step_rad_ = std::max(0.01, search_yaw_step_rad_);
    camera_height_m_ = std::max(0.01, camera_height_m_);
    initial_side_min_v_fraction_ = std::clamp(initial_side_min_v_fraction_, 0.5, 0.9);
    initial_side_max_angle_rad_ = std::clamp(initial_side_max_angle_rad_, 0.05, 0.7);
    initial_side_min_length_px_ = std::max(20.0, initial_side_min_length_px_);
    initial_side_min_confidence_ = std::clamp(initial_side_min_confidence_, 0.05, 1.0);
    initial_side_required_frames_ = std::max(1, initial_side_required_frames_);
    full_map_min_vio_translation_m_ = std::max(0.0, full_map_min_vio_translation_m_);
    full_map_min_vio_rotation_rad_ = std::max(0.0, full_map_min_vio_rotation_rad_);
  }

  void rebuildCourtMap()
  {
    court_lines_.clear();
    matching_lines_.clear();
    const double half_length = court_length_m_ * 0.5;
    const double half_width = court_width_m_ * 0.5;
    const double half_singles = singles_width_m_ * 0.5;
    const double service_x = service_line_distance_from_net_m_;

    addLine(
      -half_length, -half_width, half_length, -half_width,
      match_doubles_sidelines_);
    addLine(
      -half_length, half_width, half_length, half_width,
      match_doubles_sidelines_);
    addLine(-half_length, -half_width, -half_length, half_width);
    addLine(half_length, -half_width, half_length, half_width);
    addLine(-half_length, -half_singles, half_length, -half_singles);
    addLine(-half_length, half_singles, half_length, half_singles);
    addLine(-service_x, -half_singles, -service_x, half_singles);
    addLine(service_x, -half_singles, service_x, half_singles);
    addLine(-service_x, 0.0, service_x, 0.0);
    addLine(0.0, -half_width, 0.0, half_width);

    initial_matching_lines_left_ = {
      {{-half_length, -half_width}, {-half_length, half_width}},
      {{-half_length, half_singles}, {half_length, half_singles}},
    };
    initial_matching_lines_right_ = {
      {{-half_length, -half_width}, {-half_length, half_width}},
      {{-half_length, -half_singles}, {half_length, -half_singles}},
    };
  }

  void addLine(
    double ax,
    double ay,
    double bx,
    double by,
    bool use_for_matching = true)
  {
    const Line2 line{{ax, ay}, {bx, by}};
    court_lines_.push_back(line);
    if (use_for_matching) {
      matching_lines_.push_back(line);
    }
  }

  void resetCandidates()
  {
    candidates_.clear();
    if (start_side_ == "unknown" || start_side_ == "sideline_left") {
      candidates_.push_back(makeStartCandidate("sideline_left"));
    }
    if (start_side_ == "unknown" || start_side_ == "sideline_right") {
      candidates_.push_back(makeStartCandidate("sideline_right"));
    }
  }

  CandidatePose makeStartCandidate(const std::string & side) const
  {
    const double half_length = court_length_m_ * 0.5;
    const double half_doubles_width = court_width_m_ * 0.5;
    CandidatePose pose;
    pose.label = side;

    // Court x is the long axis and y is the short axis.  The vehicle starts
    // parallel to a short baseline at its intersection with the doubles
    // sideline.  The doubles sideline is too close to appear in the camera;
    // the farther singles sideline supplies the left/right visual cue.
    // A 180-degree rotation about the court center maps each physical corner
    // onto an indistinguishable counterpart.
    pose.x = -half_length;
    if (side == "sideline_left") {
      pose.y = half_doubles_width;
      pose.yaw = -kPi * 0.5;
    } else {
      pose.y = -half_doubles_width;
      pose.yaw = kPi * 0.5;
    }
    return pose;
  }

  void handleCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    if (msg->k[0] <= 1e-6 || msg->k[4] <= 1e-6) {
      camera_info_valid_ = false;
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        3000,
        "camera_info has invalid intrinsics; debug image and court map will still publish");
      return;
    }
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    camera_info_valid_ = true;
  }

  void handleVioOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;
    const double yaw = yawFromQuaternion(q.x, q.y, q.z, q.w);

    if (!last_vio_pose_) {
      last_vio_pose_ = CandidatePose{x, y, yaw, 0.0, "vio"};
      return;
    }

    const double dx = x - last_vio_pose_->x;
    const double dy = y - last_vio_pose_->y;
    const double dyaw = normalizeAngle(yaw - last_vio_pose_->yaw);
    accumulated_vio_translation_m_ += std::hypot(dx, dy);
    accumulated_vio_rotation_rad_ += std::abs(dyaw);
    if (
      !use_full_court_map_ &&
      (accumulated_vio_translation_m_ >= full_map_min_vio_translation_m_ ||
      accumulated_vio_rotation_rad_ >= full_map_min_vio_rotation_rad_))
    {
      use_full_court_map_ = true;
      RCLCPP_INFO(
        get_logger(),
        "switching to full court-line map after VIO motion "
        "(translation=%.2fm rotation=%.1fdeg, doubles_sidelines=%s)",
        accumulated_vio_translation_m_,
        accumulated_vio_rotation_rad_ * 180.0 / kPi,
        match_doubles_sidelines_ ? "enabled" : "disabled");
    }
    for (auto & candidate : candidates_) {
      const double c = std::cos(candidate.yaw);
      const double s = std::sin(candidate.yaw);
      candidate.x += c * dx - s * dy;
      candidate.y += s * dx + c * dy;
      candidate.yaw = normalizeAngle(candidate.yaw + dyaw);
    }
    last_vio_pose_ = CandidatePose{x, y, yaw, 0.0, "vio"};
  }

  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      const cv::Mat mono = imageToMono(*msg);
      const auto image_lines = detectImageLines(mono);
      updateInitialSideHypothesis(image_lines, mono.cols, mono.rows);
      const auto ground_lines = projectLinesToGround(image_lines);
      publishDebugImage(*msg, mono, image_lines, ground_lines.size());
      publishCourtMap();

      if (!camera_info_valid_) {
        return;
      }
      const auto best_pose = estimatePose(ground_lines);
      if (best_pose) {
        publishPose(*best_pose, msg->header.stamp);
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "court line processing failed: %s", e.what());
    }
  }

  cv::Mat imageToMono(const sensor_msgs::msg::Image & msg)
  {
    if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
      throw std::runtime_error("empty image");
    }
    const auto view = imageView(msg);
    cv::Mat mono;
    if (msg.encoding == "mono8" || msg.encoding == "8UC1") {
      mono = view.clone();
    } else if (msg.encoding == "bgr8") {
      cv::cvtColor(view, mono, cv::COLOR_BGR2GRAY);
    } else if (msg.encoding == "rgb8") {
      cv::cvtColor(view, mono, cv::COLOR_RGB2GRAY);
    } else {
      throw std::runtime_error("unsupported image encoding: " + msg.encoding);
    }
    return mono;
  }

  cv::Mat imageView(const sensor_msgs::msg::Image & msg) const
  {
    int channels = 1;
    if (msg.encoding == "bgr8" || msg.encoding == "rgb8") {
      channels = 3;
    } else if (msg.encoding != "mono8" && msg.encoding != "8UC1") {
      throw std::runtime_error("unsupported image encoding: " + msg.encoding);
    }
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

  std::vector<cv::Vec4i> detectImageLines(const cv::Mat & mono) const
  {
    const int roi_y = static_cast<int>(std::round(mono.rows * roi_start_fraction_));
    const cv::Rect roi_rect(0, roi_y, mono.cols, mono.rows - roi_y);
    const cv::Mat roi = mono(roi_rect);

    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 0.0);

    cv::Mat thresholded;
    cv::adaptiveThreshold(
      blurred,
      thresholded,
      255,
      cv::ADAPTIVE_THRESH_GAUSSIAN_C,
      cv::THRESH_BINARY,
      adaptive_block_size_,
      adaptive_c_);

    cv::Mat cleaned;
    const auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresholded, cleaned, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);

    cv::Mat edges;
    cv::Canny(cleaned, edges, 60, 160);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(
      edges,
      lines,
      1.0,
      kPi / 180.0,
      hough_threshold_,
      min_hough_line_length_px_,
      max_hough_line_gap_px_);

    for (auto & line : lines) {
      line[1] += roi_y;
      line[3] += roi_y;
    }

    std::sort(lines.begin(), lines.end(), [](const auto & lhs, const auto & rhs) {
      const double ll = std::hypot(lhs[2] - lhs[0], lhs[3] - lhs[1]);
      const double rr = std::hypot(rhs[2] - rhs[0], rhs[3] - rhs[1]);
      return ll > rr;
    });
    if (static_cast<int>(lines.size()) > max_detected_lines_) {
      lines.resize(static_cast<size_t>(max_detected_lines_));
    }
    return lines;
  }

  void updateInitialSideHypothesis(
    const std::vector<cv::Vec4i> & image_lines,
    int image_width,
    int image_height)
  {
    if (start_side_ != "unknown" || candidates_.size() != 2) {
      return;
    }

    const double center_u = camera_info_valid_ ? cx_ : image_width * 0.5;
    const double min_v = image_height * initial_side_min_v_fraction_;
    const double max_slope = std::tan(initial_side_max_angle_rad_);
    double left_length = 0.0;
    double right_length = 0.0;

    for (const auto & line : image_lines) {
      const double u0 = line[0];
      const double v0 = line[1];
      const double u1 = line[2];
      const double v1 = line[3];
      const double du = std::abs(u1 - u0);
      const double dv = std::abs(v1 - v0);
      if (du < 1.0 || dv > du * max_slope || (v0 + v1) * 0.5 < min_v) {
        continue;
      }

      const double min_u = std::min(u0, u1);
      const double max_u = std::max(u0, u1);
      left_length += std::max(0.0, std::min(max_u, center_u) - min_u);
      right_length += std::max(0.0, max_u - std::max(min_u, center_u));
    }

    const double total_length = left_length + right_length;
    const double confidence =
      total_length > 1e-6 ? std::abs(right_length - left_length) / total_length : 0.0;
    if (
      total_length < initial_side_min_length_px_ ||
      confidence < initial_side_min_confidence_)
    {
      initial_side_vote_streak_ = 0;
      initial_side_last_vote_.clear();
      return;
    }

    const std::string vote =
      right_length > left_length ? "sideline_right" : "sideline_left";
    if (vote == initial_side_last_vote_) {
      ++initial_side_vote_streak_;
    } else {
      initial_side_last_vote_ = vote;
      initial_side_vote_streak_ = 1;
    }
    if (initial_side_vote_streak_ < initial_side_required_frames_) {
      return;
    }

    candidates_.erase(
      std::remove_if(
        candidates_.begin(),
        candidates_.end(),
        [&vote](const CandidatePose & candidate) {return candidate.label != vote;}),
      candidates_.end());
    RCLCPP_INFO(
      get_logger(),
      "initial side locked to %s after %d frames "
      "(left_length=%.1fpx right_length=%.1fpx confidence=%.2f)",
      vote.c_str(),
      initial_side_vote_streak_,
      left_length,
      right_length,
      confidence);
  }

  std::vector<Line2> projectLinesToGround(const std::vector<cv::Vec4i> & image_lines) const
  {
    std::vector<Line2> ground_lines;
    if (!camera_info_valid_) {
      return ground_lines;
    }

    for (const auto & line : image_lines) {
      const auto a = projectPixelToGround(line[0], line[1]);
      const auto b = projectPixelToGround(line[2], line[3]);
      if (!a || !b) {
        continue;
      }
      if (pointDistance(*a, *b) < min_projected_line_length_m_) {
        continue;
      }
      ground_lines.push_back({*a, *b});
    }
    return ground_lines;
  }

  std::optional<Point2> projectPixelToGround(double u, double v) const
  {
    const double x_cam = (u - cx_) / fx_;
    const double y_cam = (v - cy_) / fy_;
    const double z_cam = 1.0;

    const double pitch = camera_pitch_rad_;
    const double base_x = z_cam * std::cos(pitch) - y_cam * std::sin(pitch);
    const double base_y = -x_cam;
    const double base_z = -z_cam * std::sin(pitch) - y_cam * std::cos(pitch);
    if (base_z >= -1e-4) {
      return std::nullopt;
    }
    const double t = camera_height_m_ / -base_z;
    if (!std::isfinite(t) || t <= 0.0 || t > 80.0) {
      return std::nullopt;
    }
    return Point2{base_x * t, base_y * t};
  }

  std::optional<CandidatePose> estimatePose(const std::vector<Line2> & ground_lines)
  {
    if (ground_lines.size() < 2 || candidates_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "not enough projected court lines for pose estimate: %zu",
        ground_lines.size());
      return std::nullopt;
    }

    CandidatePose best;
    for (auto & candidate : candidates_) {
      const CandidatePose refined = refineCandidate(candidate, ground_lines);
      candidate = refined;
      if (refined.score < best.score) {
        best = refined;
      }
    }

    std::sort(candidates_.begin(), candidates_.end(), [](const auto & lhs, const auto & rhs) {
      return lhs.score < rhs.score;
    });
    if (candidates_.size() > 4) {
      candidates_.resize(4);
    }

    if (best.score > match_max_average_error_m_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "court line match is weak: score=%.3fm projected_lines=%zu",
        best.score,
        ground_lines.size());
      return std::nullopt;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "court pose candidate=%s score=%.3fm x=%.2f y=%.2f yaw=%.2fdeg",
      best.label.c_str(),
      best.score,
      best.x,
      best.y,
      best.yaw * 180.0 / kPi);
    return best;
  }

  CandidatePose refineCandidate(const CandidatePose & seed, const std::vector<Line2> & ground_lines) const
  {
    CandidatePose best = seed;
    best.score = scorePose(seed, ground_lines);

    for (double dx = -search_xy_range_m_; dx <= search_xy_range_m_ + 1e-9; dx += search_xy_step_m_) {
      for (double dy = -search_xy_range_m_; dy <= search_xy_range_m_ + 1e-9; dy += search_xy_step_m_) {
        for (
          double dyaw = -search_yaw_range_rad_; dyaw <= search_yaw_range_rad_ + 1e-9;
          dyaw += search_yaw_step_rad_)
        {
          CandidatePose pose = seed;
          pose.x += dx;
          pose.y += dy;
          pose.yaw = normalizeAngle(seed.yaw + dyaw);
          pose.score = scorePose(pose, ground_lines);
          if (pose.score < best.score) {
            best = pose;
          }
        }
      }
    }
    return best;
  }

  double scorePose(const CandidatePose & pose, const std::vector<Line2> & ground_lines) const
  {
    double total = 0.0;
    int samples = 0;
    for (const auto & line : ground_lines) {
      const Point2 a = transformPoint(line.a, pose);
      const Point2 b = transformPoint(line.b, pose);
      const Point2 mid{(a.x + b.x) * 0.5, (a.y + b.y) * 0.5};
      total += nearestMapDistance(a, pose.label);
      total += nearestMapDistance(mid, pose.label);
      total += nearestMapDistance(b, pose.label);
      samples += 3;
    }
    if (samples == 0) {
      return std::numeric_limits<double>::infinity();
    }
    return total / samples;
  }

  double nearestMapDistance(const Point2 & point, const std::string & candidate_label) const
  {
    double best = std::numeric_limits<double>::infinity();
    const std::vector<Line2> * map_lines = &matching_lines_;
    if (!use_full_court_map_ && candidate_label == "sideline_left") {
      map_lines = &initial_matching_lines_left_;
    } else if (!use_full_court_map_ && candidate_label == "sideline_right") {
      map_lines = &initial_matching_lines_right_;
    }
    for (const auto & map_line : *map_lines) {
      best = std::min(best, distancePointToSegment(point, map_line));
    }
    return best;
  }

  void publishPose(const CandidatePose & pose, const builtin_interfaces::msg::Time & stamp)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = court_frame_;
    msg.pose.pose.position.x = pose.x;
    msg.pose.pose.position.y = pose.y;
    msg.pose.pose.position.z = 0.0;
    msg.pose.pose.orientation.z = std::sin(pose.yaw * 0.5);
    msg.pose.pose.orientation.w = std::cos(pose.yaw * 0.5);

    const double xy_var = std::max(0.05, pose.score * pose.score);
    const double yaw_var = std::max(0.02, pose.score * 0.25);
    msg.pose.covariance[0] = xy_var;
    msg.pose.covariance[7] = xy_var;
    msg.pose.covariance[14] = 0.25;
    msg.pose.covariance[21] = 0.25;
    msg.pose.covariance[28] = 0.25;
    msg.pose.covariance[35] = yaw_var;
    pose_pub_->publish(std::move(msg));
  }

  void publishCourtMap()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = court_frame_;
    marker.ns = "court_map";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.04;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 0.8;

    for (const auto & line : court_lines_) {
      marker.points.push_back(makePoint(line.a.x, line.a.y, 0.01));
      marker.points.push_back(makePoint(line.b.x, line.b.y, 0.01));
    }
    map_marker_pub_->publish(std::move(marker));
  }

  void publishDebugImage(
    const sensor_msgs::msg::Image & source,
    const cv::Mat & mono,
    const std::vector<cv::Vec4i> & image_lines,
    size_t projected_count)
  {
    cv::Mat debug;
    cv::cvtColor(mono, debug, cv::COLOR_GRAY2BGR);
    const int roi_y = static_cast<int>(std::round(mono.rows * roi_start_fraction_));
    cv::line(debug, cv::Point(0, roi_y), cv::Point(debug.cols - 1, roi_y), cv::Scalar(255, 200, 0), 2);
    for (const auto & line : image_lines) {
      cv::line(
        debug,
        cv::Point(line[0], line[1]),
        cv::Point(line[2], line[3]),
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);
    }
    cv::putText(
      debug,
      "court lines: " + std::to_string(image_lines.size()) +
        " projected: " + std::to_string(projected_count),
      cv::Point(20, 35),
      cv::FONT_HERSHEY_SIMPLEX,
      0.8,
      cv::Scalar(0, 255, 255),
      2);

    auto msg = sensor_msgs::msg::Image();
    msg.header = source.header;
    msg.height = static_cast<uint32_t>(debug.rows);
    msg.width = static_cast<uint32_t>(debug.cols);
    msg.encoding = "bgr8";
    msg.is_bigendian = false;
    msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(debug.cols * debug.elemSize());
    const auto data_size = static_cast<size_t>(msg.step) * debug.rows;
    msg.data.resize(data_size);
    std::memcpy(msg.data.data(), debug.data, data_size);
    debug_image_pub_->publish(std::move(msg));
  }

  std::string image_topic_;
  std::string camera_info_topic_;
  std::string vio_odom_topic_;
  std::string base_frame_;
  std::string court_frame_;
  std::string start_side_;
  double camera_height_m_{0.14214};
  double camera_pitch_rad_{0.0};
  double court_length_m_{23.77};
  double court_width_m_{10.97};
  double singles_width_m_{8.23};
  double service_line_distance_from_net_m_{6.40};
  double roi_start_fraction_{0.45};
  int min_hough_line_length_px_{45};
  int max_hough_line_gap_px_{12};
  int hough_threshold_{40};
  int adaptive_block_size_{31};
  double adaptive_c_{-8.0};
  int max_detected_lines_{40};
  double min_projected_line_length_m_{0.20};
  double match_max_average_error_m_{0.65};
  double search_xy_range_m_{2.0};
  double search_xy_step_m_{0.25};
  double search_yaw_range_rad_{0.70};
  double search_yaw_step_rad_{0.0872664626};
  double initial_side_min_v_fraction_{0.65};
  double initial_side_max_angle_rad_{0.2617993878};
  double initial_side_min_length_px_{120.0};
  double initial_side_min_confidence_{0.35};
  int initial_side_required_frames_{3};
  int initial_side_vote_streak_{0};
  std::string initial_side_last_vote_;
  bool match_doubles_sidelines_{true};
  double full_map_min_vio_translation_m_{0.5};
  double full_map_min_vio_rotation_rad_{0.35};
  double accumulated_vio_translation_m_{0.0};
  double accumulated_vio_rotation_rad_{0.0};
  bool use_full_court_map_{false};
  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  bool camera_info_valid_{false};

  std::vector<Line2> court_lines_;
  std::vector<Line2> matching_lines_;
  std::vector<Line2> initial_matching_lines_left_;
  std::vector<Line2> initial_matching_lines_right_;
  std::vector<CandidatePose> candidates_;
  std::optional<CandidatePose> last_vio_pose_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr map_marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vio_odom_sub_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CourtLineLocalizerNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("court_line_localizer_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
