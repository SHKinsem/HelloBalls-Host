#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double normalizeAngle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double clampAbs(double value, double limit)
{
  return std::clamp(value, -std::abs(limit), std::abs(limit));
}

struct Quaternion
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double w{1.0};
};

Quaternion normalizeQuaternion(Quaternion q)
{
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (!std::isfinite(norm) || norm < 1e-12) {
    return {};
  }
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return q;
}

Quaternion multiplyQuaternion(const Quaternion & lhs, const Quaternion & rhs)
{
  return normalizeQuaternion({
    lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
    lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
    lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
    lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
  });
}

Quaternion yawQuaternion(double yaw)
{
  return {0.0, 0.0, std::sin(yaw * 0.5), std::cos(yaw * 0.5)};
}

double yawFromQuaternion(const Quaternion & input)
{
  const Quaternion q = normalizeQuaternion(input);
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

struct Vector3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Vector3 rotateVector(const Quaternion & input, const Vector3 & vector)
{
  const Quaternion q = normalizeQuaternion(input);
  const double tx = 2.0 * (q.y * vector.z - q.z * vector.y);
  const double ty = 2.0 * (q.z * vector.x - q.x * vector.z);
  const double tz = 2.0 * (q.x * vector.y - q.y * vector.x);
  return {
    vector.x + q.w * tx + (q.y * tz - q.z * ty),
    vector.y + q.w * ty + (q.z * tx - q.x * tz),
    vector.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

Vector3 inverseRotateVector(const Quaternion & q, const Vector3 & vector)
{
  return rotateVector({-q.x, -q.y, -q.z, q.w}, vector);
}

struct PlanarPose
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

double planarDistance(const PlanarPose & lhs, const PlanarPose & rhs)
{
  return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
}

PlanarPose interpolatePlanar(const PlanarPose & a, const PlanarPose & b, double ratio)
{
  ratio = std::clamp(ratio, 0.0, 1.0);
  return {
    a.x + ratio * (b.x - a.x),
    a.y + ratio * (b.y - a.y),
    normalizeAngle(a.yaw + ratio * normalizeAngle(b.yaw - a.yaw)),
  };
}

PlanarPose courtToWorldFromMeasurement(
  const PlanarPose & court_to_base,
  const PlanarPose & world_to_base)
{
  const double yaw = normalizeAngle(court_to_base.yaw - world_to_base.yaw);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return {
    court_to_base.x - (c * world_to_base.x - s * world_to_base.y),
    court_to_base.y - (s * world_to_base.x + c * world_to_base.y),
    yaw,
  };
}

PlanarPose composePlanar(const PlanarPose & parent_to_middle, const PlanarPose & middle_to_child)
{
  const double c = std::cos(parent_to_middle.yaw);
  const double s = std::sin(parent_to_middle.yaw);
  return {
    parent_to_middle.x + c * middle_to_child.x - s * middle_to_child.y,
    parent_to_middle.y + s * middle_to_child.x + c * middle_to_child.y,
    normalizeAngle(parent_to_middle.yaw + middle_to_child.yaw),
  };
}

bool finitePlanar(const PlanarPose & pose)
{
  return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.yaw);
}

}  // namespace

class CourtVioFusionNode : public rclcpp::Node
{
public:
  CourtVioFusionNode()
  : Node("court_vio_fusion_node")
  {
    vio_topic_ = declare_parameter<std::string>("vio_topic", "/vio/odometry");
    court_pose_topic_ =
      declare_parameter<std::string>("court_pose_topic", "/court/pose_measurement");
    output_topic_ =
      declare_parameter<std::string>("output_topic", "/localization/odometry");
    path_topic_ = declare_parameter<std::string>("path_topic", "/localization/path");
    vehicle_marker_topic_ =
      declare_parameter<std::string>(
      "vehicle_marker_topic",
      "/localization/vehicle_markers");
    court_frame_ = declare_parameter<std::string>("court_frame", "court");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    buffer_duration_s_ = declare_parameter<double>("buffer_duration_s", 5.0);
    max_sync_error_s_ = declare_parameter<double>("max_sync_error_s", 0.15);
    confirmation_count_ = declare_parameter<int>("confirmation_count", 3);
    confirmation_translation_m_ =
      declare_parameter<double>("confirmation_translation_m", 0.6);
    confirmation_yaw_rad_ = declare_parameter<double>("confirmation_yaw_rad", 0.25);
    max_correction_jump_m_ = declare_parameter<double>("max_correction_jump_m", 1.0);
    max_correction_jump_yaw_rad_ =
      declare_parameter<double>("max_correction_jump_yaw_rad", 0.35);
    correction_gain_ = declare_parameter<double>("correction_gain", 0.2);
    max_correction_step_m_ = declare_parameter<double>("max_correction_step_m", 0.15);
    max_correction_step_yaw_rad_ =
      declare_parameter<double>("max_correction_step_yaw_rad", 0.05);
    max_white_xy_variance_ = declare_parameter<double>("max_white_xy_variance", 1.0);
    max_white_yaw_variance_ = declare_parameter<double>("max_white_yaw_variance", 0.5);
    max_path_poses_ = declare_parameter<int>("max_path_poses", 2000);
    path_min_translation_m_ =
      declare_parameter<double>("path_min_translation_m", 0.02);
    vehicle_length_m_ = declare_parameter<double>("vehicle_length_m", 0.45);
    vehicle_width_m_ = declare_parameter<double>("vehicle_width_m", 0.35);
    vehicle_height_m_ = declare_parameter<double>("vehicle_height_m", 0.18);

    validateParameters();

    output_pub_ = create_publisher<nav_msgs::msg::Odometry>(output_topic_, 50);
    path_pub_ = create_publisher<nav_msgs::msg::Path>(
      path_topic_,
      rclcpp::QoS(1).transient_local());
    vehicle_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      vehicle_marker_topic_,
      rclcpp::QoS(1).transient_local());
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    vio_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      vio_topic_,
      rclcpp::QoS(200),
      std::bind(&CourtVioFusionNode::handleVio, this, std::placeholders::_1));
    court_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      court_pose_topic_,
      rclcpp::QoS(20),
      std::bind(&CourtVioFusionNode::handleCourtPose, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "court/VIO fusion listening vio=%s court_pose=%s output=%s TF=%s->%s",
      vio_topic_.c_str(),
      court_pose_topic_.c_str(),
      output_topic_.c_str(),
      court_frame_.c_str(),
      world_frame_.c_str());
  }

private:
  struct VioSample
  {
    rclcpp::Time stamp;
    PlanarPose pose;
  };

  void validateParameters()
  {
    buffer_duration_s_ = std::max(1.0, buffer_duration_s_);
    max_sync_error_s_ = std::max(0.001, max_sync_error_s_);
    confirmation_count_ = std::max(1, confirmation_count_);
    confirmation_translation_m_ = std::max(0.01, confirmation_translation_m_);
    confirmation_yaw_rad_ = std::max(0.01, confirmation_yaw_rad_);
    max_correction_jump_m_ = std::max(confirmation_translation_m_, max_correction_jump_m_);
    max_correction_jump_yaw_rad_ =
      std::max(confirmation_yaw_rad_, max_correction_jump_yaw_rad_);
    correction_gain_ = std::clamp(correction_gain_, 0.001, 1.0);
    max_correction_step_m_ = std::max(0.001, max_correction_step_m_);
    max_correction_step_yaw_rad_ = std::max(0.001, max_correction_step_yaw_rad_);
    max_white_xy_variance_ = std::max(0.0, max_white_xy_variance_);
    max_white_yaw_variance_ = std::max(0.0, max_white_yaw_variance_);
    max_path_poses_ = std::max(10, max_path_poses_);
    path_min_translation_m_ = std::max(0.0, path_min_translation_m_);
    vehicle_length_m_ = std::max(0.05, vehicle_length_m_);
    vehicle_width_m_ = std::max(0.05, vehicle_width_m_);
    vehicle_height_m_ = std::max(0.02, vehicle_height_m_);
    if (court_frame_ == world_frame_) {
      throw std::runtime_error("court_frame and world_frame must differ");
    }
  }

  void handleVio(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != world_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "dropping VIO odometry in frame '%s'; expected '%s'",
        msg->header.frame_id.c_str(),
        world_frame_.c_str());
      return;
    }
    const rclcpp::Time stamp(msg->header.stamp);
    if (stamp.nanoseconds() <= 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping VIO odometry with zero timestamp");
      return;
    }
    if (!vio_buffer_.empty() && stamp <= vio_buffer_.back().stamp) {
      const double rewind_s = (vio_buffer_.back().stamp - stamp).seconds();
      if (rewind_s > 1.0) {
        RCLCPP_WARN(
          get_logger(),
          "odometry timestamp rewound by %.3fs; starting a new fusion run",
          rewind_s);
        resetForNewTimeline();
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "dropping out-of-order VIO odometry");
        return;
      }
    }

    if (!vio_buffer_.empty() && stamp <= vio_buffer_.back().stamp) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping out-of-order VIO odometry");
      return;
    }

    const Quaternion orientation{
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w,
    };
    const PlanarPose pose{
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      yawFromQuaternion(orientation),
    };
    if (!finitePlanar(pose)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping non-finite VIO odometry");
      return;
    }

    vio_buffer_.push_back({stamp, pose});
    const int64_t oldest_ns =
      stamp.nanoseconds() - static_cast<int64_t>(buffer_duration_s_ * 1e9);
    while (!vio_buffer_.empty() && vio_buffer_.front().stamp.nanoseconds() < oldest_ns) {
      vio_buffer_.pop_front();
    }

    if (correction_initialized_) {
      publishFusedOdometry(*msg, pose, orientation);
    }
  }

  void handleCourtPose(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
  {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != court_frame_) {
      rejectMeasurement("measurement frame does not match court_frame");
      return;
    }
    const rclcpp::Time stamp(msg->header.stamp);
    if (stamp.nanoseconds() <= 0) {
      rejectMeasurement("zero timestamp");
      return;
    }
    if (
      !vio_buffer_.empty() &&
      (vio_buffer_.back().stamp - stamp).seconds() > 1.0)
    {
      RCLCPP_WARN(
        get_logger(),
        "court measurement timestamp rewound; starting a new fusion run");
      resetForNewTimeline();
    }
    const double xy_variance =
      std::max(msg->pose.covariance[0], msg->pose.covariance[7]);
    const double yaw_variance = msg->pose.covariance[35];
    if (
      !std::isfinite(xy_variance) || !std::isfinite(yaw_variance) ||
      xy_variance < 0.0 || yaw_variance < 0.0 ||
      xy_variance > max_white_xy_variance_ ||
      yaw_variance > max_white_yaw_variance_)
    {
      rejectMeasurement("invalid or excessive covariance");
      return;
    }

    const auto vio_pose = interpolateVioPose(stamp);
    if (!vio_pose) {
      rejectMeasurement("no time-aligned VIO pose");
      return;
    }
    const Quaternion white_orientation{
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w,
    };
    const PlanarPose court_to_base{
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      yawFromQuaternion(white_orientation),
    };
    const PlanarPose candidate =
      courtToWorldFromMeasurement(court_to_base, *vio_pose);
    if (!finitePlanar(candidate)) {
      rejectMeasurement("non-finite correction candidate");
      return;
    }
    if (
      correction_initialized_ &&
      (planarDistance(candidate, correction_) > max_correction_jump_m_ ||
      std::abs(normalizeAngle(candidate.yaw - correction_.yaw)) >
      max_correction_jump_yaw_rad_))
    {
      pending_count_ = 0;
      rejectMeasurement("correction innovation exceeds gate");
      return;
    }

    accumulateConfirmedCandidate(candidate, xy_variance, yaw_variance);
  }

  std::optional<PlanarPose> interpolateVioPose(const rclcpp::Time & stamp) const
  {
    if (vio_buffer_.empty()) {
      return std::nullopt;
    }
    auto upper = std::lower_bound(
      vio_buffer_.begin(),
      vio_buffer_.end(),
      stamp,
      [](const VioSample & sample, const rclcpp::Time & target) {
        return sample.stamp < target;
      });

    if (upper == vio_buffer_.begin()) {
      const double error = std::abs((upper->stamp - stamp).seconds());
      return error <= max_sync_error_s_ ?
             std::optional<PlanarPose>(upper->pose) : std::nullopt;
    }
    if (upper == vio_buffer_.end()) {
      const auto & last = vio_buffer_.back();
      const double error = std::abs((last.stamp - stamp).seconds());
      return error <= max_sync_error_s_ ?
             std::optional<PlanarPose>(last.pose) : std::nullopt;
    }

    const auto & before = *(upper - 1);
    const double span = (upper->stamp - before.stamp).seconds();
    if (span <= 0.0 || span > 2.0 * max_sync_error_s_) {
      return std::nullopt;
    }
    const double ratio = (stamp - before.stamp).seconds() / span;
    return interpolatePlanar(before.pose, upper->pose, ratio);
  }

  void accumulateConfirmedCandidate(
    const PlanarPose & candidate,
    double xy_variance,
    double yaw_variance)
  {
    if (
      pending_count_ == 0 ||
      planarDistance(candidate, pending_correction_) > confirmation_translation_m_ ||
      std::abs(normalizeAngle(candidate.yaw - pending_correction_.yaw)) >
      confirmation_yaw_rad_)
    {
      pending_correction_ = candidate;
      pending_xy_variance_ = xy_variance;
      pending_yaw_variance_ = yaw_variance;
      pending_count_ = 1;
      return;
    }

    const double weight = 1.0 / static_cast<double>(pending_count_ + 1);
    pending_correction_.x += weight * (candidate.x - pending_correction_.x);
    pending_correction_.y += weight * (candidate.y - pending_correction_.y);
    pending_correction_.yaw = normalizeAngle(
      pending_correction_.yaw +
      weight * normalizeAngle(candidate.yaw - pending_correction_.yaw));
    pending_xy_variance_ += weight * (xy_variance - pending_xy_variance_);
    pending_yaw_variance_ += weight * (yaw_variance - pending_yaw_variance_);
    ++pending_count_;

    if (pending_count_ < confirmation_count_) {
      return;
    }

    if (!correction_initialized_) {
      correction_ = pending_correction_;
      correction_initialized_ = true;
      correction_xy_variance_ = pending_xy_variance_;
      correction_yaw_variance_ = pending_yaw_variance_;
      RCLCPP_INFO(
        get_logger(),
        "court->world initialized after %d confirmed white-line measurements: "
        "x=%.2f y=%.2f yaw=%.1fdeg",
        pending_count_,
        correction_.x,
        correction_.y,
        correction_.yaw * 180.0 / kPi);
    } else {
      applyCorrectionUpdate(
        pending_correction_,
        pending_xy_variance_,
        pending_yaw_variance_);
    }
    pending_count_ = 0;
  }

  void applyCorrectionUpdate(
    const PlanarPose & target,
    double xy_variance,
    double yaw_variance)
  {
    double dx = correction_gain_ * (target.x - correction_.x);
    double dy = correction_gain_ * (target.y - correction_.y);
    const double distance = std::hypot(dx, dy);
    if (distance > max_correction_step_m_) {
      const double scale = max_correction_step_m_ / distance;
      dx *= scale;
      dy *= scale;
    }
    const double dyaw = clampAbs(
      correction_gain_ * normalizeAngle(target.yaw - correction_.yaw),
      max_correction_step_yaw_rad_);
    correction_.x += dx;
    correction_.y += dy;
    correction_.yaw = normalizeAngle(correction_.yaw + dyaw);
    correction_xy_variance_ =
      (1.0 - correction_gain_) * correction_xy_variance_ +
      correction_gain_ * xy_variance;
    correction_yaw_variance_ =
      (1.0 - correction_gain_) * correction_yaw_variance_ +
      correction_gain_ * yaw_variance;
  }

  void publishFusedOdometry(
    const nav_msgs::msg::Odometry & vio,
    const PlanarPose & vio_planar,
    const Quaternion & vio_orientation)
  {
    const PlanarPose fused_planar = composePlanar(correction_, vio_planar);
    const Quaternion correction_q = yawQuaternion(correction_.yaw);
    const Quaternion fused_q = multiplyQuaternion(correction_q, vio_orientation);

    nav_msgs::msg::Odometry output = vio;
    output.header.frame_id = court_frame_;
    output.child_frame_id = base_frame_;
    output.pose.pose.position.x = fused_planar.x;
    output.pose.pose.position.y = fused_planar.y;
    output.pose.pose.orientation.x = fused_q.x;
    output.pose.pose.orientation.y = fused_q.y;
    output.pose.pose.orientation.z = fused_q.z;
    output.pose.pose.orientation.w = fused_q.w;
    output.pose.covariance[0] =
      std::max(output.pose.covariance[0], correction_xy_variance_);
    output.pose.covariance[7] =
      std::max(output.pose.covariance[7], correction_xy_variance_);
    output.pose.covariance[35] =
      std::max(output.pose.covariance[35], correction_yaw_variance_);

    // VINS publishes velocity in its world frame.  Odometry twist is expected
    // in child_frame_id, so rotate it back into the vehicle body frame.
    const Vector3 world_velocity{
      vio.twist.twist.linear.x,
      vio.twist.twist.linear.y,
      vio.twist.twist.linear.z,
    };
    const Vector3 body_velocity = inverseRotateVector(vio_orientation, world_velocity);
    output.twist.twist.linear.x = body_velocity.x;
    output.twist.twist.linear.y = body_velocity.y;
    output.twist.twist.linear.z = body_velocity.z;
    output_pub_->publish(output);
    publishVisualization(output);

    if (tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped transform;
      transform.header.stamp = vio.header.stamp;
      transform.header.frame_id = court_frame_;
      transform.child_frame_id = world_frame_;
      transform.transform.translation.x = correction_.x;
      transform.transform.translation.y = correction_.y;
      transform.transform.translation.z = 0.0;
      const Quaternion q = yawQuaternion(correction_.yaw);
      transform.transform.rotation.x = q.x;
      transform.transform.rotation.y = q.y;
      transform.transform.rotation.z = q.z;
      transform.transform.rotation.w = q.w;
      tf_broadcaster_->sendTransform(transform);
    }
  }

  void publishVisualization(const nav_msgs::msg::Odometry & odometry)
  {
    const auto & position = odometry.pose.pose.position;
    const bool append_path =
      path_.poses.empty() ||
      std::hypot(
      position.x - path_.poses.back().pose.position.x,
      position.y - path_.poses.back().pose.position.y) >= path_min_translation_m_;
    if (append_path) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = odometry.header;
      pose.pose = odometry.pose.pose;
      path_.poses.push_back(std::move(pose));
      while (static_cast<int>(path_.poses.size()) > max_path_poses_) {
        path_.poses.erase(path_.poses.begin());
      }
    }
    path_.header = odometry.header;
    path_pub_->publish(path_);

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker body;
    body.header = odometry.header;
    body.ns = "estimated_vehicle";
    body.id = 0;
    body.type = visualization_msgs::msg::Marker::CUBE;
    body.action = visualization_msgs::msg::Marker::ADD;
    body.pose = odometry.pose.pose;
    body.pose.position.z = vehicle_height_m_ * 0.5;
    body.scale.x = vehicle_length_m_;
    body.scale.y = vehicle_width_m_;
    body.scale.z = vehicle_height_m_;
    body.color.r = 1.0f;
    body.color.g = 0.35f;
    body.color.b = 0.05f;
    body.color.a = 0.90f;
    markers.markers.push_back(body);

    visualization_msgs::msg::Marker heading;
    heading.header = odometry.header;
    heading.ns = "estimated_vehicle";
    heading.id = 1;
    heading.type = visualization_msgs::msg::Marker::ARROW;
    heading.action = visualization_msgs::msg::Marker::ADD;
    heading.pose = odometry.pose.pose;
    heading.pose.position.z = vehicle_height_m_ + 0.03;
    heading.scale.x = vehicle_length_m_ * 1.25;
    heading.scale.y = vehicle_width_m_ * 0.22;
    heading.scale.z = vehicle_width_m_ * 0.22;
    heading.color.r = 0.10f;
    heading.color.g = 1.0f;
    heading.color.b = 0.25f;
    heading.color.a = 1.0f;
    markers.markers.push_back(heading);
    vehicle_marker_pub_->publish(markers);
  }

  void rejectMeasurement(const char * reason)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "rejecting court pose measurement: %s",
      reason);
  }

  void resetForNewTimeline()
  {
    vio_buffer_.clear();
    correction_ = PlanarPose{};
    correction_initialized_ = false;
    correction_xy_variance_ = 0.1;
    correction_yaw_variance_ = 0.1;
    pending_correction_ = PlanarPose{};
    pending_count_ = 0;
    pending_xy_variance_ = 0.0;
    pending_yaw_variance_ = 0.0;
    path_ = nav_msgs::msg::Path{};
  }

  std::string vio_topic_;
  std::string court_pose_topic_;
  std::string output_topic_;
  std::string path_topic_;
  std::string vehicle_marker_topic_;
  std::string court_frame_;
  std::string world_frame_;
  std::string base_frame_;
  bool publish_tf_{true};
  double buffer_duration_s_{5.0};
  double max_sync_error_s_{0.15};
  int confirmation_count_{3};
  double confirmation_translation_m_{0.6};
  double confirmation_yaw_rad_{0.25};
  double max_correction_jump_m_{1.0};
  double max_correction_jump_yaw_rad_{0.35};
  double correction_gain_{0.2};
  double max_correction_step_m_{0.15};
  double max_correction_step_yaw_rad_{0.05};
  double max_white_xy_variance_{1.0};
  double max_white_yaw_variance_{0.5};
  int max_path_poses_{2000};
  double path_min_translation_m_{0.02};
  double vehicle_length_m_{0.45};
  double vehicle_width_m_{0.35};
  double vehicle_height_m_{0.18};

  std::deque<VioSample> vio_buffer_;
  PlanarPose correction_;
  bool correction_initialized_{false};
  double correction_xy_variance_{0.1};
  double correction_yaw_variance_{0.1};
  PlanarPose pending_correction_;
  int pending_count_{0};
  double pending_xy_variance_{0.0};
  double pending_yaw_variance_{0.0};
  nav_msgs::msg::Path path_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr output_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    vehicle_marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vio_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    court_pose_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CourtVioFusionNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("court_vio_fusion_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
