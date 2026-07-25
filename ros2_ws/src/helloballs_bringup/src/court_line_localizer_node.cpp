#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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
#include <sensor_msgs/msg/imu.hpp>
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

struct MotionEstimate
{
  double dx{0.0};
  double dy{0.0};
  double dyaw{0.0};
  double score{std::numeric_limits<double>::infinity()};
  double inlier_ratio{0.0};
};

struct MotionPrior
{
  MotionEstimate motion;
  bool imu_yaw_valid{false};
};

struct TimedYawRate
{
  double stamp_s{0.0};
  double rate_rad_s{0.0};
};

double median(std::vector<double> values)
{
  if (values.empty()) {
    return 0.0;
  }
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if ((values.size() & 1U) != 0U) {
    return upper;
  }
  std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
  return 0.5 * (values[middle - 1] + upper);
}

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

CandidatePose composePose(const CandidatePose & pose, const MotionEstimate & motion)
{
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  CandidatePose result = pose;
  result.x += c * motion.dx - s * motion.dy;
  result.y += s * motion.dx + c * motion.dy;
  result.yaw = normalizeAngle(pose.yaw + motion.dyaw);
  return result;
}

MotionEstimate composeMotion(const MotionEstimate & first, const MotionEstimate & second)
{
  const double c = std::cos(first.dyaw);
  const double s = std::sin(first.dyaw);
  MotionEstimate result;
  result.dx = first.dx + c * second.dx - s * second.dy;
  result.dy = first.dy + s * second.dx + c * second.dy;
  result.dyaw = normalizeAngle(first.dyaw + second.dyaw);
  result.score = std::max(first.score, second.score);
  result.inlier_ratio = std::min(first.inlier_ratio, second.inlier_ratio);
  return result;
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

Point2 closestPointOnSegment(const Point2 & p, const Line2 & line)
{
  const double vx = line.b.x - line.a.x;
  const double vy = line.b.y - line.a.y;
  const double len2 = vx * vx + vy * vy;
  if (len2 <= 1e-9) {
    return line.a;
  }
  const double t = std::clamp(
    ((p.x - line.a.x) * vx + (p.y - line.a.y) * vy) / len2,
    0.0,
    1.0);
  return {line.a.x + t * vx, line.a.y + t * vy};
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
    visual_odom_topic_ =
      declare_parameter<std::string>("visual_odom_topic", "/court/visual_odometry");
    visual_odom_frame_ =
      declare_parameter<std::string>("visual_odom_frame", "world");
    use_vio_prediction_ = declare_parameter<bool>("use_vio_prediction", false);
    wheel_odom_topic_ =
      declare_parameter<std::string>("wheel_odom_topic", "/wheel/odom");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data_raw");
    use_wheel_prediction_ = declare_parameter<bool>("use_wheel_prediction", false);
    use_imu_yaw_prediction_ = declare_parameter<bool>("use_imu_yaw_prediction", false);
    imu_auto_bias_ = declare_parameter<bool>("imu_auto_bias", true);
    imu_bias_calibration_s_ = declare_parameter<double>("imu_bias_calibration_s", 2.0);
    imu_yaw_bias_rad_s_ = declare_parameter<double>("imu_yaw_bias_rad_s", 0.0);
    imu_yaw_sign_ = declare_parameter<double>("imu_yaw_sign", 1.0);
    imu_max_sample_gap_s_ = declare_parameter<double>("imu_max_sample_gap_s", 0.05);
    imu_min_coverage_ratio_ = declare_parameter<double>("imu_min_coverage_ratio", 0.70);
    camera_calibration_file_ =
      declare_parameter<std::string>("camera_calibration_file", "");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    court_frame_ = declare_parameter<std::string>("court_frame", "court");
    start_side_ = declare_parameter<std::string>("start_side", "unknown");
    camera_height_m_ = declare_parameter<double>("camera_height_m", 0.14214);
    camera_pitch_rad_ = declare_parameter<double>("camera_pitch_rad", 0.0);
    camera_offset_x_m_ = declare_parameter<double>("camera_offset_x_m", 0.23668);
    camera_offset_y_m_ = declare_parameter<double>("camera_offset_y_m", 0.0);
    court_length_m_ = declare_parameter<double>("court_length_m", 23.77);
    court_width_m_ = declare_parameter<double>("court_width_m", 10.97);
    singles_width_m_ = declare_parameter<double>("singles_width_m", 8.23);
    service_line_distance_from_net_m_ =
      declare_parameter<double>("service_line_distance_from_net_m", 6.40);
    roi_start_fraction_ = declare_parameter<double>("roi_start_fraction", 0.45);
    flow_roi_start_fraction_ =
      declare_parameter<double>("flow_roi_start_fraction", 0.60);
    flow_max_ground_range_m_ =
      declare_parameter<double>("flow_max_ground_range_m", 4.0);
    flow_max_forward_backward_error_px_ =
      declare_parameter<double>("flow_max_forward_backward_error_px", 1.5);
    enforce_nonholonomic_motion_ =
      declare_parameter<bool>("enforce_nonholonomic_motion", true);
    min_hough_line_length_px_ = declare_parameter<int>("min_hough_line_length_px", 45);
    max_hough_line_gap_px_ = declare_parameter<int>("max_hough_line_gap_px", 12);
    hough_threshold_ = declare_parameter<int>("hough_threshold", 40);
    adaptive_block_size_ = declare_parameter<int>("adaptive_block_size", 31);
    adaptive_c_ = declare_parameter<double>("adaptive_c", -8.0);
    max_detected_lines_ = declare_parameter<int>("max_detected_lines", 40);
    min_projected_line_length_m_ = declare_parameter<double>("min_projected_line_length_m", 0.20);
    match_max_average_error_m_ = declare_parameter<double>("match_max_average_error_m", 0.65);
    map_match_max_line_angle_rad_ =
      declare_parameter<double>("map_match_max_line_angle_rad", 0.30);
    map_match_orientation_weight_m_ =
      declare_parameter<double>("map_match_orientation_weight_m", 0.20);
    search_xy_range_m_ = declare_parameter<double>("search_xy_range_m", 2.0);
    search_xy_step_m_ = declare_parameter<double>("search_xy_step_m", 0.25);
    search_yaw_range_rad_ = declare_parameter<double>("search_yaw_range_rad", 0.70);
    search_yaw_step_rad_ = declare_parameter<double>("search_yaw_step_rad", 0.0872664626);
    tracking_search_xy_range_m_ =
      declare_parameter<double>("tracking_search_xy_range_m", 0.50);
    tracking_search_yaw_range_rad_ =
      declare_parameter<double>("tracking_search_yaw_range_rad", 0.20);
    initial_side_min_v_fraction_ =
      declare_parameter<double>("initial_side_min_v_fraction", 0.65);
    initial_side_max_angle_rad_ =
      declare_parameter<double>("initial_side_max_angle_rad", 0.2617993878);
    initial_side_min_length_px_ =
      declare_parameter<double>("initial_side_min_length_px", 120.0);
    initial_side_min_confidence_ =
      declare_parameter<double>("initial_side_min_confidence", 0.35);
    initial_side_required_frames_ =
      declare_parameter<int>("initial_side_required_frames", 8);
    match_doubles_sidelines_ =
      declare_parameter<bool>("match_doubles_sidelines", true);
    court_pose_update_rate_hz_ =
      declare_parameter<double>("court_pose_update_rate_hz", 4.0);
    full_map_min_vio_translation_m_ =
      declare_parameter<double>("full_map_min_vio_translation_m", 0.5);
    full_map_min_vio_rotation_rad_ =
      declare_parameter<double>("full_map_min_vio_rotation_rad", 0.35);
    frame_motion_min_lines_ = declare_parameter<int>("frame_motion_min_lines", 2);
    frame_motion_samples_per_line_ =
      declare_parameter<int>("frame_motion_samples_per_line", 7);
    frame_motion_max_gap_s_ = declare_parameter<double>("frame_motion_max_gap_s", 0.5);
    frame_motion_translation_range_m_ =
      declare_parameter<double>("frame_motion_translation_range_m", 0.60);
    frame_motion_yaw_range_rad_ =
      declare_parameter<double>("frame_motion_yaw_range_rad", 0.25);
    frame_motion_max_average_error_m_ =
      declare_parameter<double>("frame_motion_max_average_error_m", 0.20);
    frame_motion_inlier_distance_m_ =
      declare_parameter<double>("frame_motion_inlier_distance_m", 0.20);
    frame_motion_min_inlier_ratio_ =
      declare_parameter<double>("frame_motion_min_inlier_ratio", 0.45);
    frame_motion_max_line_angle_rad_ =
      declare_parameter<double>("frame_motion_max_line_angle_rad", 0.35);

    validateParameters();
    loadCameraCalibration();
    rebuildCourtMap();
    resetCandidates();

    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>("/court/debug_image", 10);
    map_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/court/map_lines", 1);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/court/pose_measurement",
      10);
    visual_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(visual_odom_topic_, 20);

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&CourtLineLocalizerNode::handleCameraInfo, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&CourtLineLocalizerNode::handleImage, this, std::placeholders::_1));
    if (use_vio_prediction_) {
      vio_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        vio_odom_topic_,
        50,
        std::bind(&CourtLineLocalizerNode::handleVioOdometry, this, std::placeholders::_1));
    }
    if (use_wheel_prediction_) {
      wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        wheel_odom_topic_,
        50,
        std::bind(&CourtLineLocalizerNode::handleWheelOdometry, this, std::placeholders::_1));
    }
    if (use_imu_yaw_prediction_) {
      imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      rclcpp::SubscriptionOptions imu_options;
      imu_options.callback_group = imu_callback_group_;
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_,
        rclcpp::SensorDataQoS().keep_last(200),
        std::bind(&CourtLineLocalizerNode::handleImu, this, std::placeholders::_1),
        imu_options);
    }

    map_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&CourtLineLocalizerNode::publishCourtMap, this));

    RCLCPP_INFO(
      get_logger(),
      "court line localizer listening image=%s camera_info=%s visual_odom=%s "
      "external_vio=%s wheel=%s imu_yaw=%s start_side=%s",
      image_topic_.c_str(),
      camera_info_topic_.c_str(),
      visual_odom_topic_.c_str(),
      use_vio_prediction_ ? vio_odom_topic_.c_str() : "disabled",
      use_wheel_prediction_ ? wheel_odom_topic_.c_str() : "disabled",
      use_imu_yaw_prediction_ ? imu_topic_.c_str() : "disabled",
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
    flow_roi_start_fraction_ = std::clamp(flow_roi_start_fraction_, 0.45, 0.9);
    flow_max_ground_range_m_ = std::max(0.5, flow_max_ground_range_m_);
    flow_max_forward_backward_error_px_ =
      std::max(0.1, flow_max_forward_backward_error_px_);
    max_detected_lines_ = std::max(1, max_detected_lines_);
    map_match_max_line_angle_rad_ =
      std::clamp(map_match_max_line_angle_rad_, 0.01, kPi * 0.5);
    map_match_orientation_weight_m_ = std::max(0.0, map_match_orientation_weight_m_);
    search_xy_step_m_ = std::max(0.05, search_xy_step_m_);
    search_yaw_step_rad_ = std::max(0.01, search_yaw_step_rad_);
    tracking_search_xy_range_m_ =
      std::clamp(tracking_search_xy_range_m_, search_xy_step_m_, search_xy_range_m_);
    tracking_search_yaw_range_rad_ =
      std::clamp(
      tracking_search_yaw_range_rad_,
      search_yaw_step_rad_,
      search_yaw_range_rad_);
    camera_height_m_ = std::max(0.01, camera_height_m_);
    initial_side_min_v_fraction_ = std::clamp(initial_side_min_v_fraction_, 0.5, 0.9);
    initial_side_max_angle_rad_ = std::clamp(initial_side_max_angle_rad_, 0.05, 0.7);
    initial_side_min_length_px_ = std::max(20.0, initial_side_min_length_px_);
    initial_side_min_confidence_ = std::clamp(initial_side_min_confidence_, 0.05, 1.0);
    initial_side_required_frames_ = std::max(1, initial_side_required_frames_);
    full_map_min_vio_translation_m_ = std::max(0.0, full_map_min_vio_translation_m_);
    full_map_min_vio_rotation_rad_ = std::max(0.0, full_map_min_vio_rotation_rad_);
    court_pose_update_rate_hz_ = std::clamp(court_pose_update_rate_hz_, 0.5, 30.0);
    frame_motion_min_lines_ = std::max(1, frame_motion_min_lines_);
    frame_motion_samples_per_line_ = std::max(3, frame_motion_samples_per_line_);
    frame_motion_max_gap_s_ = std::max(0.05, frame_motion_max_gap_s_);
    frame_motion_translation_range_m_ = std::max(0.0, frame_motion_translation_range_m_);
    frame_motion_yaw_range_rad_ = std::max(0.0, frame_motion_yaw_range_rad_);
    frame_motion_max_average_error_m_ = std::max(0.01, frame_motion_max_average_error_m_);
    frame_motion_inlier_distance_m_ = std::max(0.01, frame_motion_inlier_distance_m_);
    frame_motion_min_inlier_ratio_ =
      std::clamp(frame_motion_min_inlier_ratio_, 0.05, 1.0);
    frame_motion_max_line_angle_rad_ =
      std::clamp(frame_motion_max_line_angle_rad_, 0.01, kPi * 0.5);
    imu_bias_calibration_s_ = std::max(0.2, imu_bias_calibration_s_);
    imu_yaw_sign_ = imu_yaw_sign_ < 0.0 ? -1.0 : 1.0;
    imu_max_sample_gap_s_ = std::clamp(imu_max_sample_gap_s_, 0.005, 0.5);
    imu_min_coverage_ratio_ = std::clamp(imu_min_coverage_ratio_, 0.1, 1.0);
    imu_bias_ready_ = !imu_auto_bias_;
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

  void loadCameraCalibration()
  {
    if (camera_calibration_file_.empty()) {
      return;
    }
    cv::FileStorage calibration(camera_calibration_file_, cv::FileStorage::READ);
    if (!calibration.isOpened()) {
      throw std::runtime_error(
              "failed to open camera calibration file: " + camera_calibration_file_);
    }
    const cv::FileNode projection = calibration["projection_parameters"];
    projection["fx"] >> fx_;
    projection["fy"] >> fy_;
    projection["cx"] >> cx_;
    projection["cy"] >> cy_;
    if (fx_ <= 1e-6 || fy_ <= 1e-6) {
      throw std::runtime_error(
              "camera calibration file has invalid projection parameters: " +
              camera_calibration_file_);
    }
    camera_info_valid_ = true;
    camera_info_from_file_ = true;
    RCLCPP_INFO(
      get_logger(),
      "loaded fallback camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f from %s",
      fx_,
      fy_,
      cx_,
      cy_,
      camera_calibration_file_.c_str());
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
    initial_side_votes_.clear();
    initial_side_locked_ = start_side_ != "unknown";
    court_pose_tracking_ = false;
    consecutive_court_pose_failures_ = 0;
    successful_court_pose_measurements_ = 0;
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
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        3000,
        camera_info_from_file_ ?
        "camera_info has invalid intrinsics; keeping calibration-file intrinsics" :
        "camera_info has invalid intrinsics; debug image and court map will still publish");
      if (!camera_info_from_file_) {
        camera_info_valid_ = false;
      }
      return;
    }
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    camera_info_valid_ = true;
    camera_info_from_file_ = false;
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

    const double world_dx = x - last_vio_pose_->x;
    const double world_dy = y - last_vio_pose_->y;
    const double dyaw = normalizeAngle(yaw - last_vio_pose_->yaw);
    const double c = std::cos(last_vio_pose_->yaw);
    const double s = std::sin(last_vio_pose_->yaw);
    const MotionEstimate motion{
      c * world_dx + s * world_dy,
      -s * world_dx + c * world_dy,
      dyaw,
      0.0,
      1.0};
    applyCandidateMotion(motion, "external VIO");
    last_vio_pose_ = CandidatePose{x, y, yaw, 0.0, "vio"};
  }

  void handleWheelOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const auto & position = msg->pose.pose.position;
    const auto & q = msg->pose.pose.orientation;
    const CandidatePose pose{
      position.x,
      position.y,
      yawFromQuaternion(q.x, q.y, q.z, q.w),
      0.0,
      "wheel"};
    if (!last_wheel_pose_) {
      last_wheel_pose_ = pose;
      return;
    }

    const double world_dx = pose.x - last_wheel_pose_->x;
    const double world_dy = pose.y - last_wheel_pose_->y;
    const double c = std::cos(last_wheel_pose_->yaw);
    const double s = std::sin(last_wheel_pose_->yaw);
    const MotionEstimate increment{
      c * world_dx + s * world_dy,
      -s * world_dx + c * world_dy,
      normalizeAngle(pose.yaw - last_wheel_pose_->yaw),
      0.10,
      1.0};
    pending_wheel_motion_ = composeMotion(pending_wheel_motion_, increment);
    last_wheel_pose_ = pose;
  }

  void handleImu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    const double stamp_s =
      static_cast<double>(msg->header.stamp.sec) +
      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    const double yaw_rate = imu_yaw_sign_ * msg->angular_velocity.z;
    if (
      !std::isfinite(stamp_s) || stamp_s <= 0.0 ||
      !std::isfinite(yaw_rate))
    {
      return;
    }

    std::lock_guard<std::mutex> lock(imu_mutex_);
    if (!imu_samples_.empty() && stamp_s <= imu_samples_.back().stamp_s) {
      const double rewind_s = imu_samples_.back().stamp_s - stamp_s;
      if (rewind_s <= 1.0) {
        return;
      }
      RCLCPP_WARN(
        get_logger(),
        "IMU timestamp rewound by %.3fs; resetting gyro timeline",
        rewind_s);
      imu_samples_.clear();
      imu_bias_samples_.clear();
      imu_bias_start_stamp_s_.reset();
      imu_bias_ready_ = !imu_auto_bias_;
    }
    imu_samples_.push_back({stamp_s, yaw_rate});
    while (!imu_samples_.empty() && stamp_s - imu_samples_.front().stamp_s > 10.0) {
      imu_samples_.pop_front();
    }

    if (imu_auto_bias_ && !imu_bias_ready_) {
      if (!imu_bias_start_stamp_s_) {
        imu_bias_start_stamp_s_ = stamp_s;
      }
      imu_bias_samples_.push_back(yaw_rate);
      if (stamp_s - *imu_bias_start_stamp_s_ >= imu_bias_calibration_s_) {
        imu_yaw_bias_rad_s_ = median(imu_bias_samples_);
        imu_bias_ready_ = true;
        RCLCPP_INFO(
          get_logger(),
          "IMU yaw bias calibrated from %zu samples: %.6f rad/s",
          imu_bias_samples_.size(),
          imu_yaw_bias_rad_s_);
        imu_bias_samples_.clear();
      }
    }
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

      const double stamp_s =
        static_cast<double>(msg->header.stamp.sec) +
        static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
      updateFrameMotion(mono, ground_lines, stamp_s, msg->header.stamp);

      const bool update_court_pose =
        !last_court_pose_update_stamp_s_ ||
        stamp_s - *last_court_pose_update_stamp_s_ >=
        1.0 / court_pose_update_rate_hz_;
      if (update_court_pose) {
        last_court_pose_update_stamp_s_ = stamp_s;
        const auto best_pose = estimatePose(ground_lines);
        if (best_pose) {
          publishPose(*best_pose, msg->header.stamp);
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "court line processing failed: %s", e.what());
    }
  }

  void updateFrameMotion(
    const cv::Mat & mono,
    const std::vector<Line2> & ground_lines,
    double stamp_s,
    const builtin_interfaces::msg::Time & stamp)
  {
    if (!std::isfinite(stamp_s) || stamp_s <= 0.0) {
      previous_ground_lines_.clear();
      previous_image_stamp_s_.reset();
      return;
    }

    if (!previous_image_stamp_s_) {
      previous_ground_lines_ = ground_lines;
      previous_mono_ = mono.clone();
      previous_image_stamp_s_ = stamp_s;
      visual_odom_pose_ = CandidatePose{};
      clearAuxiliaryMotion();
      publishVisualOdometry(stamp, MotionEstimate{}, 0.0, false);
      return;
    }

    const double dt = stamp_s - *previous_image_stamp_s_;
    if (dt <= 0.0) {
      if (dt < -1.0) {
        RCLCPP_WARN(
          get_logger(),
          "image timestamp rewound by %.3fs; starting a new localization run",
          -dt);
        resetForNewTimeline();
        previous_ground_lines_ = ground_lines;
        previous_mono_ = mono.clone();
        previous_image_stamp_s_ = stamp_s;
        publishVisualOdometry(stamp, MotionEstimate{}, 0.0, false);
        return;
      }
      RCLCPP_WARN(
        get_logger(),
        "resetting frame motion across non-increasing image timestamp gap %.3fs",
        dt);
      previous_ground_lines_ = ground_lines;
      previous_mono_ = mono.clone();
      previous_image_stamp_s_ = stamp_s;
      clearAuxiliaryMotion();
      return;
    }

    const auto prior = auxiliaryMotion(*previous_image_stamp_s_, stamp_s);
    if (dt > frame_motion_max_gap_s_) {
      // A visual keyframe gap does not imply an IMU gap. Preserve the gyro
      // heading increment, then reset only the optical-flow reference.
      if (prior && prior->imu_yaw_valid) {
        MotionEstimate yaw_only = prior->motion;
        if (!use_wheel_prediction_) {
          yaw_only.dx = 0.0;
          yaw_only.dy = 0.0;
        }
        yaw_only.score = frame_motion_max_average_error_m_;
        yaw_only.inlier_ratio = 0.0;
        applyCandidateMotion(yaw_only, "IMU bridge across image gap");
        visual_odom_pose_ = composePose(visual_odom_pose_, yaw_only);
        publishVisualOdometry(stamp, yaw_only, dt, true);
      } else {
        RCLCPP_WARN(
          get_logger(),
          "resetting visual motion across %.3fs gap without sufficient IMU coverage",
          dt);
      }
      previous_ground_lines_ = ground_lines;
      previous_mono_ = mono.clone();
      previous_image_stamp_s_ = stamp_s;
      clearAuxiliaryMotion();
      return;
    }

    auto motion = estimateImageFrameMotion(previous_mono_, mono, prior);
    if (!motion) {
      motion = estimateFrameMotion(previous_ground_lines_, ground_lines, prior);
    }
    if (motion) {
      applyCandidateMotion(*motion, "white-line frame registration");
      visual_odom_pose_ = composePose(visual_odom_pose_, *motion);
      publishVisualOdometry(stamp, *motion, dt, true);
      consecutive_frame_motion_failures_ = 0;
    } else if (prior && prior->imu_yaw_valid) {
      // Even when image translation is temporarily unobservable, gyro yaw is
      // still a valid planar increment. Publish it and keep the optical-flow
      // reference adjacent so a failed frame cannot hide a whole turn.
      MotionEstimate yaw_only = prior->motion;
      if (!use_wheel_prediction_) {
        yaw_only.dx = 0.0;
        yaw_only.dy = 0.0;
      }
      yaw_only.score = frame_motion_max_average_error_m_;
      yaw_only.inlier_ratio = 0.0;
      applyCandidateMotion(yaw_only, "IMU yaw fallback");
      visual_odom_pose_ = composePose(visual_odom_pose_, yaw_only);
      publishVisualOdometry(stamp, yaw_only, dt, true);
      ++consecutive_frame_motion_failures_;
    } else {
      ++consecutive_frame_motion_failures_;
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "white-line frame motion rejected (%d consecutive frames)",
        consecutive_frame_motion_failures_);
      // Keep the last accepted frame as the keyframe. Advancing it here would
      // permanently discard the unestimated displacement and systematically
      // shorten the travelled distance.
      return;
    }

    previous_ground_lines_ = ground_lines;
    previous_mono_ = mono.clone();
    previous_image_stamp_s_ = stamp_s;
    clearAuxiliaryMotion();
  }

  void resetForNewTimeline()
  {
    previous_ground_lines_.clear();
    previous_mono_.release();
    previous_image_stamp_s_.reset();
    last_court_pose_update_stamp_s_.reset();
    visual_odom_pose_ = CandidatePose{};
    last_vio_pose_.reset();
    last_wheel_pose_.reset();
    pending_wheel_motion_ = MotionEstimate{};
    accumulated_vio_translation_m_ = 0.0;
    accumulated_vio_rotation_rad_ = 0.0;
    use_full_court_map_ = false;
    consecutive_frame_motion_failures_ = 0;
    resetCandidates();
  }

  std::optional<MotionEstimate> estimateImageFrameMotion(
    const cv::Mat & previous,
    const cv::Mat & current,
    const std::optional<MotionPrior> & prior) const
  {
    if (
      previous.empty() || current.empty() || previous.size() != current.size() ||
      !camera_info_valid_)
    {
      return std::nullopt;
    }

    cv::Mat feature_mask = cv::Mat::zeros(previous.size(), CV_8UC1);
    const int roi_y =
      static_cast<int>(std::round(previous.rows * flow_roi_start_fraction_));
    feature_mask(cv::Rect(0, roi_y, previous.cols, previous.rows - roi_y)).setTo(255);

    std::vector<cv::Point2f> previous_pixels;
    cv::goodFeaturesToTrack(
      previous,
      previous_pixels,
      240,
      0.01,
      8.0,
      feature_mask,
      5,
      false,
      0.04);
    const int minimum_tracks = prior && prior->imu_yaw_valid ? 6 : 10;
    if (static_cast<int>(previous_pixels.size()) < minimum_tracks) {
      return std::nullopt;
    }

    std::vector<cv::Point2f> current_pixels;
    std::vector<unsigned char> status;
    std::vector<float> tracking_error;
    cv::calcOpticalFlowPyrLK(
      previous,
      current,
      previous_pixels,
      current_pixels,
      status,
      tracking_error,
      cv::Size(21, 21),
      3,
      cv::TermCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        30,
        0.01));

    std::vector<cv::Point2f> backward_pixels;
    std::vector<unsigned char> backward_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
      current,
      previous,
      current_pixels,
      backward_pixels,
      backward_status,
      backward_error,
      cv::Size(21, 21),
      3,
      cv::TermCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        30,
        0.01));

    std::vector<cv::Point2f> current_ground;
    std::vector<cv::Point2f> previous_ground;
    for (size_t i = 0; i < previous_pixels.size(); ++i) {
      if (
        !status[i] || !backward_status[i] ||
        tracking_error[i] > (prior && prior->imu_yaw_valid ? 50.0f : 30.0f) ||
        cv::norm(backward_pixels[i] - previous_pixels[i]) >
        flow_max_forward_backward_error_px_ ||
        current_pixels[i].x < 0.0f || current_pixels[i].x >= current.cols ||
        current_pixels[i].y < roi_y || current_pixels[i].y >= current.rows)
      {
        continue;
      }
      const auto before = projectPixelToGround(previous_pixels[i].x, previous_pixels[i].y);
      const auto after = projectPixelToGround(current_pixels[i].x, current_pixels[i].y);
      if (!before || !after) {
        continue;
      }
      if (
        std::hypot(before->x, before->y) > flow_max_ground_range_m_ ||
        std::hypot(after->x, after->y) > flow_max_ground_range_m_)
      {
        continue;
      }
      previous_ground.emplace_back(
        static_cast<float>(before->x),
        static_cast<float>(before->y));
      current_ground.emplace_back(
        static_cast<float>(after->x),
        static_cast<float>(after->y));
    }
    if (static_cast<int>(current_ground.size()) < minimum_tracks) {
      return std::nullopt;
    }

    MotionEstimate motion;
    cv::Mat inlier_mask;
    if (prior && prior->imu_yaw_valid) {
      // Rotation around the camera makes planar optical flow poorly
      // conditioned when most detected features lie on one long white line.
      // Use the bias-corrected gyro increment for rotation, then robustly solve
      // only the two translation components from all tracked ground points.
      motion.dyaw = prior->motion.dyaw;
      const double c = std::cos(motion.dyaw);
      const double s = std::sin(motion.dyaw);
      std::vector<double> translations_x;
      std::vector<double> translations_y;
      translations_x.reserve(current_ground.size());
      translations_y.reserve(current_ground.size());
      for (size_t i = 0; i < current_ground.size(); ++i) {
        const auto & point = current_ground[i];
        const auto & target = previous_ground[i];
        translations_x.push_back(target.x - (c * point.x - s * point.y));
        translations_y.push_back(target.y - (s * point.x + c * point.y));
      }
      motion.dx = median(translations_x);
      motion.dy = median(translations_y);

      inlier_mask = cv::Mat::zeros(
        static_cast<int>(current_ground.size()), 1, CV_8UC1);
      translations_x.clear();
      translations_y.clear();
      for (size_t i = 0; i < current_ground.size(); ++i) {
        const auto & point = current_ground[i];
        const auto & target = previous_ground[i];
        const double ex = motion.dx + c * point.x - s * point.y - target.x;
        const double ey = motion.dy + s * point.x + c * point.y - target.y;
        if (std::hypot(ex, ey) <= frame_motion_inlier_distance_m_) {
          inlier_mask.at<unsigned char>(static_cast<int>(i), 0) = 1;
          translations_x.push_back(target.x - (c * point.x - s * point.y));
          translations_y.push_back(target.y - (s * point.x + c * point.y));
        }
      }
      if (static_cast<int>(translations_x.size()) >= minimum_tracks) {
        motion.dx = median(translations_x);
        motion.dy = median(translations_y);
      }
    } else {
      const cv::Mat affine = cv::estimateAffinePartial2D(
        current_ground,
        previous_ground,
        inlier_mask,
        cv::RANSAC,
        frame_motion_inlier_distance_m_,
        1000,
        0.995,
        10);
      if (affine.empty()) {
        return std::nullopt;
      }

      const double a = affine.at<double>(0, 0);
      const double b = affine.at<double>(1, 0);
      const double scale = std::hypot(a, b);
      if (!std::isfinite(scale) || std::abs(scale - 1.0) > 0.10) {
        return std::nullopt;
      }
      motion.dx = affine.at<double>(0, 2);
      motion.dy = affine.at<double>(1, 2);
      motion.dyaw = std::atan2(b, a);
    }

    int inliers = 0;
    double squared_error = 0.0;
    const double c = std::cos(motion.dyaw);
    const double s = std::sin(motion.dyaw);
    for (int i = 0; i < inlier_mask.rows; ++i) {
      if (inlier_mask.at<unsigned char>(i, 0) == 0) {
        continue;
      }
      ++inliers;
      const auto & point = current_ground[static_cast<size_t>(i)];
      const auto & target = previous_ground[static_cast<size_t>(i)];
      const double predicted_x = motion.dx + c * point.x - s * point.y;
      const double predicted_y = motion.dy + s * point.x + c * point.y;
      const double ex = predicted_x - target.x;
      const double ey = predicted_y - target.y;
      squared_error += ex * ex + ey * ey;
    }
    motion.inlier_ratio =
      static_cast<double>(inliers) / static_cast<double>(current_ground.size());
    motion.score =
      inliers > 0 ? std::sqrt(squared_error / static_cast<double>(inliers)) :
      std::numeric_limits<double>::infinity();
    const int minimum_inliers = prior && prior->imu_yaw_valid ? 5 : 10;
    const double minimum_inlier_ratio =
      prior && prior->imu_yaw_valid ? 0.25 : frame_motion_min_inlier_ratio_;
    const double maximum_error =
      prior && prior->imu_yaw_valid ? 0.30 : frame_motion_max_average_error_m_;
    if (
      inliers < minimum_inliers ||
      motion.inlier_ratio < minimum_inlier_ratio ||
      motion.score > maximum_error)
    {
      return std::nullopt;
    }

    MotionEstimate base_motion = cameraMotionToBase(motion);
    if (enforce_nonholonomic_motion_) {
      base_motion = projectToNonholonomicMotion(base_motion);
    }
    const MotionEstimate center = prior ? prior->motion : MotionEstimate{};
    if (
      std::hypot(base_motion.dx - center.dx, base_motion.dy - center.dy) >
      frame_motion_translation_range_m_ ||
      std::abs(normalizeAngle(base_motion.dyaw - center.dyaw)) >
      frame_motion_yaw_range_rad_)
    {
      return std::nullopt;
    }
    return base_motion;
  }

  std::optional<MotionEstimate> estimateFrameMotion(
    const std::vector<Line2> & previous_lines,
    const std::vector<Line2> & current_lines,
    const std::optional<MotionPrior> & prior) const
  {
    if (
      static_cast<int>(previous_lines.size()) < frame_motion_min_lines_ ||
      static_cast<int>(current_lines.size()) < frame_motion_min_lines_ ||
      !hasOrientationDiversity(previous_lines) ||
      !hasOrientationDiversity(current_lines))
    {
      return std::nullopt;
    }

    const MotionEstimate base_center = prior ? prior->motion : MotionEstimate{};
    const MotionEstimate center = baseMotionToCamera(base_center);
    MotionEstimate estimate = center;
    for (int iteration = 0; iteration < 5; ++iteration) {
      std::vector<cv::Point2f> source_points;
      std::vector<cv::Point2f> target_points;
      const double c = std::cos(estimate.dyaw);
      const double s = std::sin(estimate.dyaw);

      for (const auto & current : current_lines) {
        const double current_angle =
          std::atan2(current.b.y - current.a.y, current.b.x - current.a.x) +
          estimate.dyaw;
        for (int i = 0; i < frame_motion_samples_per_line_; ++i) {
          const double ratio =
            static_cast<double>(i) /
            static_cast<double>(frame_motion_samples_per_line_ - 1);
          const Point2 source{
            current.a.x + ratio * (current.b.x - current.a.x),
            current.a.y + ratio * (current.b.y - current.a.y)};
          const Point2 transformed{
            estimate.dx + c * source.x - s * source.y,
            estimate.dy + s * source.x + c * source.y};

          double nearest_distance = frame_motion_max_average_error_m_ * 3.0;
          std::optional<Point2> nearest_point;
          for (const auto & previous : previous_lines) {
            const double previous_angle =
              std::atan2(previous.b.y - previous.a.y, previous.b.x - previous.a.x);
            double angle_error = std::abs(normalizeAngle(current_angle - previous_angle));
            angle_error = std::min(angle_error, std::abs(kPi - angle_error));
            if (angle_error > frame_motion_max_line_angle_rad_) {
              continue;
            }
            const Point2 projected = closestPointOnSegment(transformed, previous);
            const double distance = pointDistance(transformed, projected);
            if (distance < nearest_distance) {
              nearest_distance = distance;
              nearest_point = projected;
            }
          }
          if (nearest_point) {
            source_points.emplace_back(
              static_cast<float>(source.x),
              static_cast<float>(source.y));
            target_points.emplace_back(
              static_cast<float>(nearest_point->x),
              static_cast<float>(nearest_point->y));
          }
        }
      }

      if (source_points.size() < 6) {
        return std::nullopt;
      }
      cv::Mat inlier_mask;
      const cv::Mat affine = cv::estimateAffinePartial2D(
        source_points,
        target_points,
        inlier_mask,
        cv::RANSAC,
        frame_motion_inlier_distance_m_,
        500,
        0.99,
        10);
      if (affine.empty()) {
        return std::nullopt;
      }

      const double a = affine.at<double>(0, 0);
      const double b = affine.at<double>(1, 0);
      const double scale = std::hypot(a, b);
      if (!std::isfinite(scale) || std::abs(scale - 1.0) > 0.15) {
        return std::nullopt;
      }
      MotionEstimate next;
      next.dx = affine.at<double>(0, 2);
      next.dy = affine.at<double>(1, 2);
      next.dyaw = std::atan2(b, a);
      if (
        std::hypot(next.dx - center.dx, next.dy - center.dy) >
        frame_motion_translation_range_m_ ||
        std::abs(normalizeAngle(next.dyaw - center.dyaw)) >
        frame_motion_yaw_range_rad_)
      {
        return std::nullopt;
      }
      estimate = next;
    }

    const auto [score, inlier_ratio] =
      scoreFrameMotion(previous_lines, current_lines, estimate);
    estimate.score = score;
    estimate.inlier_ratio = inlier_ratio;
    if (
      estimate.score > frame_motion_max_average_error_m_ ||
      estimate.inlier_ratio < frame_motion_min_inlier_ratio_)
    {
      return std::nullopt;
    }
    MotionEstimate base_motion = cameraMotionToBase(estimate);
    return enforce_nonholonomic_motion_ ?
           projectToNonholonomicMotion(base_motion) : base_motion;
  }

  MotionEstimate projectToNonholonomicMotion(const MotionEstimate & input) const
  {
    MotionEstimate output = input;
    double arc_x = 1.0;
    double arc_y = 0.0;
    if (std::abs(input.dyaw) > 1e-5) {
      arc_x = std::sin(input.dyaw) / input.dyaw;
      arc_y = (1.0 - std::cos(input.dyaw)) / input.dyaw;
    }
    const double norm2 = arc_x * arc_x + arc_y * arc_y;
    const double distance =
      norm2 > 1e-9 ? (input.dx * arc_x + input.dy * arc_y) / norm2 : 0.0;
    output.dx = distance * arc_x;
    output.dy = distance * arc_y;
    return output;
  }

  MotionEstimate cameraMotionToBase(const MotionEstimate & camera_motion) const
  {
    MotionEstimate base_motion = camera_motion;
    const double c = std::cos(camera_motion.dyaw);
    const double s = std::sin(camera_motion.dyaw);
    base_motion.dx +=
      camera_offset_x_m_ -
      (c * camera_offset_x_m_ - s * camera_offset_y_m_);
    base_motion.dy +=
      camera_offset_y_m_ -
      (s * camera_offset_x_m_ + c * camera_offset_y_m_);
    return base_motion;
  }

  MotionEstimate baseMotionToCamera(const MotionEstimate & base_motion) const
  {
    MotionEstimate camera_motion = base_motion;
    const double c = std::cos(base_motion.dyaw);
    const double s = std::sin(base_motion.dyaw);
    camera_motion.dx +=
      c * camera_offset_x_m_ - s * camera_offset_y_m_ - camera_offset_x_m_;
    camera_motion.dy +=
      s * camera_offset_x_m_ + c * camera_offset_y_m_ - camera_offset_y_m_;
    return camera_motion;
  }

  std::optional<double> integrateImuYaw(double start_stamp_s, double end_stamp_s)
  {
    const double duration = end_stamp_s - start_stamp_s;
    if (!use_imu_yaw_prediction_ || duration <= 0.0) {
      return std::nullopt;
    }

    std::deque<TimedYawRate> samples;
    double bias = 0.0;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      if (!imu_bias_ready_) {
        return std::nullopt;
      }
      samples = imu_samples_;
      bias = imu_yaw_bias_rad_s_;
    }
    if (samples.size() < 2) {
      return std::nullopt;
    }

    double yaw = 0.0;
    double covered_s = 0.0;
    for (size_t i = 1; i < samples.size(); ++i) {
      const auto & first = samples[i - 1];
      const auto & second = samples[i];
      const double sample_dt = second.stamp_s - first.stamp_s;
      if (
        sample_dt <= 0.0 || sample_dt > imu_max_sample_gap_s_ ||
        second.stamp_s <= start_stamp_s || first.stamp_s >= end_stamp_s)
      {
        continue;
      }
      const double segment_start = std::max(start_stamp_s, first.stamp_s);
      const double segment_end = std::min(end_stamp_s, second.stamp_s);
      if (segment_end <= segment_start) {
        continue;
      }
      const double start_ratio = (segment_start - first.stamp_s) / sample_dt;
      const double end_ratio = (segment_end - first.stamp_s) / sample_dt;
      const double start_rate =
        first.rate_rad_s +
        start_ratio * (second.rate_rad_s - first.rate_rad_s) - bias;
      const double end_rate =
        first.rate_rad_s +
        end_ratio * (second.rate_rad_s - first.rate_rad_s) - bias;
      const double segment_dt = segment_end - segment_start;
      yaw += 0.5 * (start_rate + end_rate) * segment_dt;
      covered_s += segment_dt;
    }
    if (covered_s < duration * imu_min_coverage_ratio_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "IMU coverage %.0f%% is insufficient for image interval %.3fs",
        100.0 * covered_s / duration,
        duration);
      return std::nullopt;
    }
    return normalizeAngle(yaw);
  }

  std::optional<MotionPrior> auxiliaryMotion(double start_stamp_s, double end_stamp_s)
  {
    const bool has_wheel =
      use_wheel_prediction_ &&
      (std::hypot(pending_wheel_motion_.dx, pending_wheel_motion_.dy) > 1e-6 ||
      std::abs(pending_wheel_motion_.dyaw) > 1e-6);
    const auto imu_yaw = integrateImuYaw(start_stamp_s, end_stamp_s);
    const bool has_imu = imu_yaw.has_value();
    if (!has_wheel && !has_imu) {
      return std::nullopt;
    }

    MotionPrior prior;
    prior.motion = has_wheel ? pending_wheel_motion_ : MotionEstimate{};
    if (has_imu) {
      prior.motion.dyaw = *imu_yaw;
      prior.imu_yaw_valid = true;
    }
    prior.motion.score = 0.10;
    prior.motion.inlier_ratio = 1.0;
    if (
      std::hypot(prior.motion.dx, prior.motion.dy) >
      frame_motion_translation_range_m_ * 2.0 ||
      std::abs(prior.motion.dyaw) > 3.0 * (end_stamp_s - start_stamp_s) + 0.05)
    {
      RCLCPP_WARN(
        get_logger(),
        "discarding implausible wheel/IMU frame prior dx=%.2f dy=%.2f dyaw=%.1fdeg",
        prior.motion.dx,
        prior.motion.dy,
        prior.motion.dyaw * 180.0 / kPi);
      return std::nullopt;
    }
    return prior;
  }

  void clearAuxiliaryMotion()
  {
    pending_wheel_motion_ = MotionEstimate{};
  }

  std::pair<double, double> scoreFrameMotion(
    const std::vector<Line2> & previous_lines,
    const std::vector<Line2> & current_lines,
    const MotionEstimate & motion) const
  {
    std::vector<double> distances;
    distances.reserve(
      current_lines.size() * static_cast<size_t>(frame_motion_samples_per_line_));
    int inliers = 0;

    const double c = std::cos(motion.dyaw);
    const double s = std::sin(motion.dyaw);
    for (const auto & current : current_lines) {
      const double current_angle =
        std::atan2(current.b.y - current.a.y, current.b.x - current.a.x) + motion.dyaw;
      for (int i = 0; i < frame_motion_samples_per_line_; ++i) {
        const double ratio =
          static_cast<double>(i) / static_cast<double>(frame_motion_samples_per_line_ - 1);
        const Point2 point{
          current.a.x + ratio * (current.b.x - current.a.x),
          current.a.y + ratio * (current.b.y - current.a.y)};
        const Point2 transformed{
          motion.dx + c * point.x - s * point.y,
          motion.dy + s * point.x + c * point.y};

        double nearest = frame_motion_max_average_error_m_ * 4.0;
        for (const auto & previous : previous_lines) {
          const double previous_angle =
            std::atan2(previous.b.y - previous.a.y, previous.b.x - previous.a.x);
          double angle_error = std::abs(normalizeAngle(current_angle - previous_angle));
          angle_error = std::min(angle_error, std::abs(kPi - angle_error));
          if (angle_error > frame_motion_max_line_angle_rad_) {
            continue;
          }
          nearest = std::min(nearest, distancePointToSegment(transformed, previous));
        }
        distances.push_back(nearest);
        if (nearest <= frame_motion_inlier_distance_m_) {
          ++inliers;
        }
      }
    }

    if (distances.empty()) {
      return {std::numeric_limits<double>::infinity(), 0.0};
    }
    std::sort(distances.begin(), distances.end());
    const size_t kept = std::max<size_t>(
      1,
      static_cast<size_t>(std::ceil(static_cast<double>(distances.size()) * 0.7)));
    double total = 0.0;
    for (size_t i = 0; i < kept; ++i) {
      total += distances[i];
    }
    return {
      total / static_cast<double>(kept),
      static_cast<double>(inliers) / static_cast<double>(distances.size())};
  }

  bool hasOrientationDiversity(const std::vector<Line2> & lines) const
  {
    for (size_t i = 0; i < lines.size(); ++i) {
      const double first =
        std::atan2(lines[i].b.y - lines[i].a.y, lines[i].b.x - lines[i].a.x);
      for (size_t j = i + 1; j < lines.size(); ++j) {
        const double second =
          std::atan2(lines[j].b.y - lines[j].a.y, lines[j].b.x - lines[j].a.x);
        double difference = std::abs(normalizeAngle(first - second));
        difference = std::min(difference, std::abs(kPi - difference));
        if (difference >= frame_motion_max_line_angle_rad_) {
          return true;
        }
      }
    }
    return false;
  }

  void applyCandidateMotion(const MotionEstimate & motion, const char * source)
  {
    for (auto & candidate : candidates_) {
      candidate = composePose(candidate, motion);
    }
    accumulated_vio_translation_m_ += std::hypot(motion.dx, motion.dy);
    accumulated_vio_rotation_rad_ += std::abs(motion.dyaw);
    if (
      initial_side_locked_ && successful_court_pose_measurements_ >= 3 &&
      !use_full_court_map_ &&
      (accumulated_vio_translation_m_ >= full_map_min_vio_translation_m_ ||
      accumulated_vio_rotation_rad_ >= full_map_min_vio_rotation_rad_))
    {
      use_full_court_map_ = true;
      RCLCPP_INFO(
        get_logger(),
        "switching to full court-line map after %s motion "
        "(translation=%.2fm rotation=%.1fdeg, doubles_sidelines=%s)",
        source,
        accumulated_vio_translation_m_,
        accumulated_vio_rotation_rad_ * 180.0 / kPi,
        match_doubles_sidelines_ ? "enabled" : "disabled");
    }
  }

  void publishVisualOdometry(
    const builtin_interfaces::msg::Time & stamp,
    const MotionEstimate & motion,
    double dt,
    bool motion_valid)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = visual_odom_frame_;
    msg.child_frame_id = base_frame_;
    msg.pose.pose.position.x = visual_odom_pose_.x;
    msg.pose.pose.position.y = visual_odom_pose_.y;
    msg.pose.pose.orientation.z = std::sin(visual_odom_pose_.yaw * 0.5);
    msg.pose.pose.orientation.w = std::cos(visual_odom_pose_.yaw * 0.5);

    const double position_variance =
      motion_valid ? std::max(0.01, motion.score * motion.score) : 1.0;
    const double yaw_variance =
      motion_valid ? std::max(0.01, motion.score * 0.2) : 0.5;
    msg.pose.covariance[0] = position_variance;
    msg.pose.covariance[7] = position_variance;
    msg.pose.covariance[35] = yaw_variance;
    if (motion_valid && dt > 1e-6) {
      msg.twist.twist.linear.x = motion.dx / dt;
      msg.twist.twist.linear.y = motion.dy / dt;
      msg.twist.twist.angular.z = motion.dyaw / dt;
    }
    visual_odom_pub_->publish(std::move(msg));
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

    // Keep the startup cue tied to the visible image halves.  The calibrated
    // principal point can be noticeably off-center and is not the semantic
    // divider used by the original left/right corner heuristic.
    const double center_u = image_width * 0.5;
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
    if (total_length < initial_side_min_length_px_) {
      return;
    }

    const double evidence = (right_length - left_length) / total_length;
    initial_side_votes_.push_back(evidence);
    const size_t vote_window = static_cast<size_t>(initial_side_required_frames_ * 2);
    while (initial_side_votes_.size() > vote_window) {
      initial_side_votes_.pop_front();
    }
    if (initial_side_votes_.size() < static_cast<size_t>(initial_side_required_frames_)) {
      return;
    }
    const double mean_evidence =
      std::accumulate(initial_side_votes_.begin(), initial_side_votes_.end(), 0.0) /
      static_cast<double>(initial_side_votes_.size());
    const double confidence = std::abs(mean_evidence);
    if (confidence < initial_side_min_confidence_) {
      return;
    }
    const std::string vote =
      mean_evidence > 0.0 ? "sideline_right" : "sideline_left";

    candidates_.erase(
      std::remove_if(
        candidates_.begin(),
        candidates_.end(),
        [&vote](const CandidatePose & candidate) {return candidate.label != vote;}),
      candidates_.end());
    initial_side_locked_ = true;
    court_pose_tracking_ = false;
    successful_court_pose_measurements_ = 0;
    RCLCPP_INFO(
      get_logger(),
      "initial side locked to %s after %d frames "
      "(left_length=%.1fpx right_length=%.1fpx confidence=%.2f)",
      vote.c_str(),
      static_cast<int>(initial_side_votes_.size()),
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
    if (!initial_side_locked_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "holding court pose until the initial side hypothesis is locked");
      return std::nullopt;
    }
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
      ++consecutive_court_pose_failures_;
      if (consecutive_court_pose_failures_ >= 10) {
        court_pose_tracking_ = false;
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "court line match is weak: score=%.3fm projected_lines=%zu",
        best.score,
        ground_lines.size());
      return std::nullopt;
    }

    consecutive_court_pose_failures_ = 0;
    court_pose_tracking_ = true;
    ++successful_court_pose_measurements_;
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

    const double xy_range =
      court_pose_tracking_ ? tracking_search_xy_range_m_ : search_xy_range_m_;
    const double yaw_range =
      court_pose_tracking_ ? tracking_search_yaw_range_rad_ : search_yaw_range_rad_;
    const double coarse_xy_step = search_xy_step_m_ * 2.0;
    const double coarse_yaw_step = search_yaw_step_rad_ * 2.0;
    searchPoseNeighborhood(
      seed,
      ground_lines,
      xy_range,
      coarse_xy_step,
      yaw_range,
      coarse_yaw_step,
      best);
    const CandidatePose coarse_best = best;
    searchPoseNeighborhood(
      coarse_best,
      ground_lines,
      coarse_xy_step,
      search_xy_step_m_,
      coarse_yaw_step,
      search_yaw_step_rad_,
      best);
    return best;
  }

  void searchPoseNeighborhood(
    const CandidatePose & center,
    const std::vector<Line2> & ground_lines,
    double xy_range,
    double xy_step,
    double yaw_range,
    double yaw_step,
    CandidatePose & best) const
  {
    for (double dx = -xy_range; dx <= xy_range + 1e-9; dx += xy_step) {
      for (double dy = -xy_range; dy <= xy_range + 1e-9; dy += xy_step) {
        for (double dyaw = -yaw_range; dyaw <= yaw_range + 1e-9; dyaw += yaw_step) {
          CandidatePose pose = center;
          pose.x += dx;
          pose.y += dy;
          pose.yaw = normalizeAngle(center.yaw + dyaw);
          pose.score = scorePose(pose, ground_lines);
          if (pose.score < best.score) {
            best = pose;
          }
        }
      }
    }
  }

  double scorePose(const CandidatePose & pose, const std::vector<Line2> & ground_lines) const
  {
    const double half_length = court_length_m_ * 0.5;
    const double half_width = court_width_m_ * 0.5;
    constexpr double kMapBoundaryMarginM = 1.0;
    if (
      std::abs(pose.x) > half_length + kMapBoundaryMarginM ||
      std::abs(pose.y) > half_width + kMapBoundaryMarginM)
    {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<double> line_scores;
    line_scores.reserve(ground_lines.size());
    for (const auto & line : ground_lines) {
      const Point2 a = transformPoint(line.a, pose);
      const Point2 b = transformPoint(line.b, pose);
      const Point2 mid{(a.x + b.x) * 0.5, (a.y + b.y) * 0.5};
      const double observed_angle = std::atan2(b.y - a.y, b.x - a.x);

      const std::vector<Line2> * map_lines = &matching_lines_;
      if (!use_full_court_map_ && pose.label == "sideline_left") {
        map_lines = &initial_matching_lines_left_;
      } else if (!use_full_court_map_ && pose.label == "sideline_right") {
        map_lines = &initial_matching_lines_right_;
      }

      double best_line_score = std::numeric_limits<double>::infinity();
      for (const auto & map_line : *map_lines) {
        const double map_angle =
          std::atan2(map_line.b.y - map_line.a.y, map_line.b.x - map_line.a.x);
        double angle_error = std::abs(normalizeAngle(observed_angle - map_angle));
        angle_error = std::min(angle_error, std::abs(kPi - angle_error));
        if (angle_error > map_match_max_line_angle_rad_) {
          continue;
        }
        const double distance =
          (distancePointToSegment(a, map_line) +
          distancePointToSegment(mid, map_line) +
          distancePointToSegment(b, map_line)) / 3.0;
        best_line_score = std::min(
          best_line_score,
          distance + map_match_orientation_weight_m_ * angle_error);
      }
      if (std::isfinite(best_line_score)) {
        line_scores.push_back(best_line_score);
      }
    }
    if (line_scores.size() < 2) {
      return std::numeric_limits<double>::infinity();
    }
    std::sort(line_scores.begin(), line_scores.end());
    const size_t kept = std::max<size_t>(
      2,
      static_cast<size_t>(std::ceil(static_cast<double>(line_scores.size()) * 0.7)));
    double total = 0.0;
    for (size_t i = 0; i < kept; ++i) {
      total += line_scores[i];
    }
    return total / static_cast<double>(kept);
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
    marker.scale.x = 0.08;
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
  std::string visual_odom_topic_;
  std::string visual_odom_frame_;
  bool use_vio_prediction_{false};
  std::string wheel_odom_topic_;
  std::string imu_topic_;
  std::string camera_calibration_file_;
  bool use_wheel_prediction_{false};
  bool use_imu_yaw_prediction_{false};
  bool imu_auto_bias_{true};
  double imu_bias_calibration_s_{2.0};
  double imu_yaw_bias_rad_s_{0.0};
  double imu_yaw_sign_{1.0};
  double imu_max_sample_gap_s_{0.05};
  double imu_min_coverage_ratio_{0.70};
  std::string base_frame_;
  std::string court_frame_;
  std::string start_side_;
  double camera_height_m_{0.14214};
  double camera_pitch_rad_{0.0};
  double camera_offset_x_m_{0.23668};
  double camera_offset_y_m_{0.0};
  double court_length_m_{23.77};
  double court_width_m_{10.97};
  double singles_width_m_{8.23};
  double service_line_distance_from_net_m_{6.40};
  double roi_start_fraction_{0.45};
  double flow_roi_start_fraction_{0.60};
  double flow_max_ground_range_m_{4.0};
  double flow_max_forward_backward_error_px_{1.5};
  bool enforce_nonholonomic_motion_{true};
  int min_hough_line_length_px_{45};
  int max_hough_line_gap_px_{12};
  int hough_threshold_{40};
  int adaptive_block_size_{31};
  double adaptive_c_{-8.0};
  int max_detected_lines_{40};
  double min_projected_line_length_m_{0.20};
  double match_max_average_error_m_{0.65};
  double map_match_max_line_angle_rad_{0.30};
  double map_match_orientation_weight_m_{0.20};
  double search_xy_range_m_{2.0};
  double search_xy_step_m_{0.25};
  double search_yaw_range_rad_{0.70};
  double search_yaw_step_rad_{0.0872664626};
  double tracking_search_xy_range_m_{0.50};
  double tracking_search_yaw_range_rad_{0.20};
  double initial_side_min_v_fraction_{0.65};
  double initial_side_max_angle_rad_{0.2617993878};
  double initial_side_min_length_px_{120.0};
  double initial_side_min_confidence_{0.35};
  int initial_side_required_frames_{3};
  std::deque<double> initial_side_votes_;
  bool match_doubles_sidelines_{true};
  double court_pose_update_rate_hz_{4.0};
  double full_map_min_vio_translation_m_{0.5};
  double full_map_min_vio_rotation_rad_{0.35};
  double accumulated_vio_translation_m_{0.0};
  double accumulated_vio_rotation_rad_{0.0};
  bool use_full_court_map_{false};
  bool initial_side_locked_{false};
  bool court_pose_tracking_{false};
  std::optional<double> last_court_pose_update_stamp_s_;
  int consecutive_court_pose_failures_{0};
  int successful_court_pose_measurements_{0};
  int frame_motion_min_lines_{2};
  int frame_motion_samples_per_line_{7};
  double frame_motion_max_gap_s_{0.5};
  double frame_motion_translation_range_m_{0.60};
  double frame_motion_yaw_range_rad_{0.25};
  double frame_motion_max_average_error_m_{0.20};
  double frame_motion_inlier_distance_m_{0.20};
  double frame_motion_min_inlier_ratio_{0.45};
  double frame_motion_max_line_angle_rad_{0.35};
  int consecutive_frame_motion_failures_{0};
  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  bool camera_info_valid_{false};
  bool camera_info_from_file_{false};

  std::vector<Line2> court_lines_;
  std::vector<Line2> matching_lines_;
  std::vector<Line2> initial_matching_lines_left_;
  std::vector<Line2> initial_matching_lines_right_;
  std::vector<CandidatePose> candidates_;
  std::optional<CandidatePose> last_vio_pose_;
  std::optional<CandidatePose> last_wheel_pose_;
  MotionEstimate pending_wheel_motion_;
  mutable std::mutex imu_mutex_;
  std::deque<TimedYawRate> imu_samples_;
  std::vector<double> imu_bias_samples_;
  std::optional<double> imu_bias_start_stamp_s_;
  bool imu_bias_ready_{false};
  std::vector<Line2> previous_ground_lines_;
  cv::Mat previous_mono_;
  std::optional<double> previous_image_stamp_s_;
  CandidatePose visual_odom_pose_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr map_marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr visual_odom_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vio_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<CourtLineLocalizerNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("court_line_localizer_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
