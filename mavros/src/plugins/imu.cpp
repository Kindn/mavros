/*
 * Copyright 2013-2017,2021 Vladimir Ermakov.
 *
 * This file is part of the mavros package and subject to the license terms
 * in the top-level LICENSE file of the mavros repository.
 * https://github.com/mavlink/mavros/tree/master/LICENSE.md
 */
/**
 * @brief IMU and attitude data parser plugin
 * @file imu.cpp
 * @author Vladimir Ermakov <vooon341@gmail.com>
 *
 * @addtogroup plugin
 * @{
 */

#include <tf2_eigen/tf2_eigen.h>

#include <cmath>
#include <string>

#include "mavros/mavros_uas.hpp"
#include "mavros/plugin.hpp"
#include "mavros/plugin_filter.hpp"
#include "rcpputils/asserts.hpp"

#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "sensor_msgs/msg/temperature.hpp"

#include "simple_json/json.hpp"

namespace mavros {
namespace std_plugins {
using namespace std::placeholders; // NOLINT

//! Gauss to Tesla coeff
static constexpr double GAUSS_TO_TESLA = 1.0e-4;
//! millTesla to Tesla coeff
static constexpr double MILLIT_TO_TESLA = 1000.0;
//! millRad/Sec to Rad/Sec coeff
static constexpr double MILLIRS_TO_RADSEC = 1.0e-3;
//! millG to m/s**2 coeff
static constexpr double MILLIG_TO_MS2 = 9.80665 / 1000.0;
//! millm/s**2 to m/s**2 coeff
static constexpr double MILLIMS2_TO_MS2 = 1.0e-3;
//! millBar to Pascal coeff
static constexpr double MILLIBAR_TO_PASCAL = 1.0e2;
//! Radians to degrees
static constexpr double RAD_TO_DEG = 180.0 / M_PI;

/**
 * @brief IMU and attitude data publication plugin
 * @plugin imu
 */
class IMUPlugin : public plugin::Plugin {
public:
  struct GyrCalibInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double g_xy_flu{0};
    double g_yx_flu{0};
    double g_yz_flu{0};
    double g_zy_flu{0};
    double g_zx_flu{0};
    double g_xz_flu{0};
    Eigen::Vector3d s_flu{Eigen::Vector3d::Ones()};
    Eigen::Vector3d b_flu{Eigen::Vector3d::Zero()};

    std::string DebugStr() const {
      std::stringstream ss{};
      ss << "\tg_xy_flu: " << g_xy_flu << "\n"
         << "\tg_yx_flu: " << g_yx_flu << "\n"
         << "\tg_yz_flu: " << g_yz_flu << "\n"
         << "\tg_zy_flu: " << g_zy_flu << "\n"
         << "\tg_zx_flu: " << g_zx_flu << "\n"
         << "\tg_xz_flu: " << g_xz_flu << "\n"
         << "\ts_flu: [" << s_flu.transpose() << "]\n"
         << "\tb_flu: [" << b_flu.transpose() << "]";

      return ss.str();
    }
  };

  struct AccCalibInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double a_yz_flu{0};
    double a_zy_flu{0};
    double a_zx_flu{0};
    Eigen::Vector3d s_flu{Eigen::Vector3d::Ones()};
    Eigen::Vector3d b_flu{Eigen::Vector3d::Zero()};

    std::string DebugStr() const {
      std::stringstream ss{};
      ss << "\ta_yz_flu: " << a_yz_flu << "\n"
         << "\ta_zy_flu: " << a_zy_flu << "\n"
         << "\ta_zx_flu: " << a_zx_flu << "\n"
         << "\ts_flu: [" << s_flu.transpose() << "]\n"
         << "\tb_flu: [" << b_flu.transpose() << "]";

      return ss.str();
    }
  };

  struct ImuCalibInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GyrCalibInfo gyro{};
    AccCalibInfo accel{};

    std::string DebugStr() const {
      std::stringstream ss{};
      ss << "IMU Calib Info --------------" << std::endl;
      ss << "gyroscope: \n"
         << gyro.DebugStr() << "\n"
         << "accelerometer: \n"
         << accel.DebugStr() << std::endl;

      return ss.str();
    }
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit IMUPlugin(plugin::UASPtr uas_)
      : Plugin(uas_, "imu"), has_hr_imu(false), has_raw_imu(false),
        has_scaled_imu(false), has_att_quat(false),
        received_linear_accel(false),
        linear_accel_vec_flu(Eigen::Vector3d::Zero()),
        linear_accel_vec_frd(Eigen::Vector3d::Zero()) {
    enable_node_watch_parameters();

    /**
     * @warning A rotation from the aircraft-frame to the base_link frame is
     * applied. Additionally, it is reported the orientation of the vehicle to
     * describe the transformation from the ENU frame to the base_link frame
     * (ENU <-> base_link). THIS ORIENTATION IS NOT THE SAME AS THAT REPORTED BY
     * THE FCU (NED <-> aircraft).
     */
    node_declare_and_watch_parameter(
        "frame_id", "base_link",
        [&](const rclcpp::Parameter &p) { frame_id = p.as_string(); });

    node_declare_and_watch_parameter(
        "linear_acceleration_stdev", 0.0003, [&](const rclcpp::Parameter &p) {
          auto linear_stdev = p.as_double();
          setup_covariance(linear_acceleration_cov, linear_stdev);
        });
    node_declare_and_watch_parameter(
        "angular_velocity_stdev", 0.02 * (M_PI / 180.0),
        [&](const rclcpp::Parameter &p) {
          auto angular_stdev = p.as_double();
          setup_covariance(angular_velocity_cov, angular_stdev);
        });
    node_declare_and_watch_parameter(
        "orientation_stdev", 1.0, [&](const rclcpp::Parameter &p) {
          auto orientation_stdev = p.as_double();
          setup_covariance(orientation_cov, orientation_stdev);
        });
    node_declare_and_watch_parameter(
        "magnetic_stdev", 0.0, [&](const rclcpp::Parameter &p) {
          auto mag_stdev = p.as_double();
          setup_covariance(magnetic_cov, mag_stdev);
        });
    node_declare_and_watch_parameter(
        "gyro_calib_info_path", "", [&](const rclcpp::Parameter &p) {
          auto logger{get_logger()};

          std::string const path{p.as_string()};
          std::ifstream ifs{path};

          sjson::Json calib_info_json{ifs};
          if (!calib_info_json.succeed()) {
            RCLCPP_WARN(logger,
                        "Failed to read gyroscope calib info from %s. Default "
                        "parameters will be used.",
                        path.c_str());
          } else {
            auto calib_info_root{calib_info_json.getRoot()};
            auto gamma{calib_info_root["gamma"]};
            imu_calib_info.gyro.g_xy_flu = gamma["xy"].as_double();
            imu_calib_info.gyro.g_yx_flu = gamma["yx"].as_double();
            imu_calib_info.gyro.g_yz_flu = gamma["yz"].as_double();
            imu_calib_info.gyro.g_zy_flu = gamma["zy"].as_double();
            imu_calib_info.gyro.g_zx_flu = gamma["zx"].as_double();
            imu_calib_info.gyro.g_xz_flu = gamma["xz"].as_double();
            auto scale{calib_info_root["scale"].as_vector()};
            imu_calib_info.gyro.s_flu.x() = scale[0]->as_double();
            imu_calib_info.gyro.s_flu.y() = scale[1]->as_double();
            imu_calib_info.gyro.s_flu.z() = scale[2]->as_double();
            auto bias{calib_info_root["bias"].as_vector()};
            imu_calib_info.gyro.b_flu.x() = bias[0]->as_double();
            imu_calib_info.gyro.b_flu.y() = bias[1]->as_double();
            imu_calib_info.gyro.b_flu.z() = bias[2]->as_double();
          }

          gyro_transform.setIdentity();
          //* Element-wise assignment instead of << operators are faster and
          //* avoid some problems
          gyro_transform(0, 1) = -imu_calib_info.gyro.g_yz_flu;
          gyro_transform(0, 2) = imu_calib_info.gyro.g_zy_flu;
          gyro_transform(1, 0) = imu_calib_info.gyro.g_xz_flu;
          gyro_transform(1, 2) = -imu_calib_info.gyro.g_zx_flu;
          gyro_transform(2, 0) = -imu_calib_info.gyro.g_xy_flu;
          gyro_transform(2, 1) = imu_calib_info.gyro.g_yx_flu;
          gyro_transform =
              gyro_transform * imu_calib_info.gyro.s_flu.asDiagonal();

          RCLCPP_INFO(logger, "gyro_calib_info: \n%s",
                      imu_calib_info.gyro.DebugStr().c_str());
        });

    node_declare_and_watch_parameter(
        "accel_calib_info_path", "", [&](const rclcpp::Parameter &p) {
          auto logger{get_logger()};

          std::string const path{p.as_string()};
          std::ifstream ifs{path};

          sjson::Json calib_info_json{ifs};
          if (!calib_info_json.succeed()) {
            RCLCPP_WARN(logger,
                        "Failed to read gyroscope calib info from %s. Default "
                        "parameters will be used.",
                        path.c_str());
          } else {
            auto calib_info_root{calib_info_json.getRoot()};
            imu_calib_info.accel.a_yz_flu =
                calib_info_root["alpha_yz"].as_double();
            imu_calib_info.accel.a_zy_flu =
                calib_info_root["alpha_zy"].as_double();
            imu_calib_info.accel.a_zx_flu =
                calib_info_root["alpha_zx"].as_double();
            auto scale{calib_info_root["scale"].as_vector()};
            imu_calib_info.accel.s_flu.x() = scale[0]->as_double();
            imu_calib_info.accel.s_flu.y() = scale[1]->as_double();
            imu_calib_info.accel.s_flu.z() = scale[2]->as_double();
            auto bias{calib_info_root["bias"].as_vector()};
            imu_calib_info.accel.b_flu.x() = bias[0]->as_double();
            imu_calib_info.accel.b_flu.y() = bias[1]->as_double();
            imu_calib_info.accel.b_flu.z() = bias[2]->as_double();
          }

          accel_transform.setIdentity();
          //* Element-wise assignment instead of << operators are faster and
          //* avoid some problems
          accel_transform(0, 1) = -imu_calib_info.accel.a_yz_flu;
          accel_transform(0, 2) = imu_calib_info.accel.a_zy_flu;
          accel_transform(1, 2) = -imu_calib_info.accel.a_zx_flu;
          accel_transform =
              accel_transform * imu_calib_info.accel.s_flu.asDiagonal();

          RCLCPP_INFO(logger, "accel_calib_info: \n%s",
                      imu_calib_info.accel.DebugStr().c_str());
        });
    node_declare_and_watch_parameter(
        "pub_data_raw", true, [&](const rclcpp::Parameter &p) {
          pub_data_raw = p.as_bool();
          if (!pub_data_raw) {
            auto logger{get_logger()};
            RCLCPP_WARN(logger, "Raw IMU data will not be published.");
          }
        });

    setup_covariance(unk_orientation_cov, 0.0);

    auto sensor_qos = rclcpp::SensorDataQoS();

    imu_pub =
        node->create_publisher<sensor_msgs::msg::Imu>("~/data", sensor_qos);
    imu_raw_pub =
        node->create_publisher<sensor_msgs::msg::Imu>("~/data_raw", sensor_qos);
    imu_corrected_pub = node->create_publisher<sensor_msgs::msg::Imu>(
        "~/data_corrected", sensor_qos);
    magn_pub = node->create_publisher<sensor_msgs::msg::MagneticField>(
        "~/mag", sensor_qos);
    temp_imu_pub = node->create_publisher<sensor_msgs::msg::Temperature>(
        "~/temperature_imu", sensor_qos);
    temp_baro_pub = node->create_publisher<sensor_msgs::msg::Temperature>(
        "~/temperature_baro", sensor_qos);
    static_press_pub = node->create_publisher<sensor_msgs::msg::FluidPressure>(
        "~/static_pressure", sensor_qos);
    diff_press_pub = node->create_publisher<sensor_msgs::msg::FluidPressure>(
        "~/diff_pressure", sensor_qos);

    // Reset has_* flags on connection change
    enable_connection_cb();
  }

  Subscriptions get_subscriptions() override {
    return {
        make_handler(&IMUPlugin::handle_attitude),
        make_handler(&IMUPlugin::handle_attitude_quaternion),
        make_handler(&IMUPlugin::handle_highres_imu),
        make_handler(&IMUPlugin::handle_raw_imu),
        make_handler(&IMUPlugin::handle_scaled_imu),
        make_handler(&IMUPlugin::handle_scaled_pressure),
    };
  }

private:
  std::string frame_id;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_raw_pub;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_corrected_pub;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr magn_pub;
  rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr temp_imu_pub;
  rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr temp_baro_pub;
  rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr
      static_press_pub;
  rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr diff_press_pub;

  std::atomic<bool> has_hr_imu;
  std::atomic<bool> has_raw_imu;
  std::atomic<bool> has_scaled_imu;
  std::atomic<bool> has_att_quat;
  std::atomic<bool> received_linear_accel;
  Eigen::Vector3d linear_accel_vec_flu;
  Eigen::Vector3d linear_accel_vec_frd;
  ftf::Covariance3d linear_acceleration_cov;
  ftf::Covariance3d angular_velocity_cov;
  ftf::Covariance3d corrected_linear_acceleration_cov;
  ftf::Covariance3d corrected_angular_velocity_cov;
  ftf::Covariance3d orientation_cov;
  ftf::Covariance3d unk_orientation_cov;
  ftf::Covariance3d magnetic_cov;
  ImuCalibInfo imu_calib_info{};
  Eigen::Matrix3d gyro_transform{};
  Eigen::Matrix3d accel_transform{};
  Eigen::Vector3d corrected_gyro_flu{};
  Eigen::Vector3d corrected_accel_flu{};
  bool pub_data_raw{true};

  /* -*- helpers -*- */

  /**
   * @brief Setup 3x3 covariance matrix
   * @param cov		Covariance matrix
   * @param stdev		Standard deviation
   * @remarks		Diagonal computed from the stdev
   */
  void setup_covariance(ftf::Covariance3d &cov, double stdev) {
    ftf::EigenMapCovariance3d c(cov.data());

    c.setZero();
    if (stdev) {
      double sr = stdev * stdev;
      c.diagonal() << sr, sr, sr;
    } else {
      c(0, 0) = -1.0;
    }
  }

  /**
   * @brief Fill and publish IMU data message.
   * @param time_boot_ms     Message timestamp (not syncronized)
   * @param orientation_enu  Orientation in the base_link ENU frame
   * @param orientation_ned  Orientation in the aircraft NED frame
   * @param gyro_flu         Angular velocity/rate in the base_link
   * Forward-Left-Up frame
   * @param gyro_frd         Angular velocity/rate in the aircraft
   * Forward-Right-Down frame
   */
  void publish_imu_data(uint32_t time_boot_ms,
                        Eigen::Quaterniond &orientation_enu,
                        Eigen::Quaterniond &orientation_ned,
                        Eigen::Vector3d &gyro_flu, Eigen::Vector3d &gyro_frd) {
    auto imu_ned_msg = sensor_msgs::msg::Imu();
    auto imu_enu_msg = sensor_msgs::msg::Imu();

    // Fill message header
    imu_enu_msg.header = uas->synchronized_header(frame_id, time_boot_ms);
    imu_ned_msg.header = uas->synchronized_header("aircraft", time_boot_ms);

    // Convert from Eigen::Quaternond to geometry_msgs::Quaternion
    imu_enu_msg.orientation = tf2::toMsg(orientation_enu);
    imu_ned_msg.orientation = tf2::toMsg(orientation_ned);

    // Convert from Eigen::Vector3d to geometry_msgs::Vector3
    tf2::toMsg(gyro_flu, imu_enu_msg.angular_velocity);
    tf2::toMsg(gyro_frd, imu_ned_msg.angular_velocity);

    // Eigen::Vector3d from HIGHRES_IMU or RAW_IMU, to geometry_msgs::Vector3
    tf2::toMsg(linear_accel_vec_flu, imu_enu_msg.linear_acceleration);
    tf2::toMsg(linear_accel_vec_frd, imu_ned_msg.linear_acceleration);

    // Pass ENU msg covariances
    imu_enu_msg.orientation_covariance = orientation_cov;
    imu_enu_msg.angular_velocity_covariance = angular_velocity_cov;
    imu_enu_msg.linear_acceleration_covariance = linear_acceleration_cov;

    // Pass NED msg covariances
    imu_ned_msg.orientation_covariance = orientation_cov;
    imu_ned_msg.angular_velocity_covariance = angular_velocity_cov;
    imu_ned_msg.linear_acceleration_covariance = linear_acceleration_cov;

    if (!received_linear_accel) {
      // Set element 0 of covariance matrix to -1
      // if no data received as per sensor_msgs/Imu defintion
      imu_enu_msg.linear_acceleration_covariance[0] = -1;
      imu_ned_msg.linear_acceleration_covariance[0] = -1;
    }

    /** Store attitude in base_link ENU
     *  @snippet src/plugins/imu.cpp store_enu
     */
    // [store_enu]
    uas->data.update_attitude_imu_enu(imu_enu_msg);
    // [store_enu]

    /** Store attitude in aircraft NED
     *  @snippet src/plugins/imu.cpp store_ned
     */
    // [store_enu]
    uas->data.update_attitude_imu_ned(imu_ned_msg);
    // [store_ned]

    /** Publish only base_link ENU message
     *  @snippet src/plugins/imu.cpp pub_enu
     */
    // [pub_enu]
    imu_pub->publish(imu_enu_msg);
    // [pub_enu]
  }

  void correct_gyro_raw(Eigen::Vector3d const &gyro_raw,
                        Eigen::Vector3d &gyro_corrected) {
    Eigen::Vector3d const gyro_wo_b{imu_calib_info.gyro.s_flu.cwiseProduct(
        gyro_raw - imu_calib_info.gyro.b_flu)};
    gyro_corrected.x() = gyro_wo_b.x() +
                         (-imu_calib_info.gyro.g_yz_flu) * gyro_wo_b.y() +
                         imu_calib_info.gyro.g_zy_flu * gyro_wo_b.z();
    gyro_corrected.y() = imu_calib_info.gyro.g_xz_flu * gyro_wo_b.x() +
                         gyro_wo_b.y() +
                         (-imu_calib_info.gyro.g_zx_flu) * gyro_wo_b.z();
    gyro_corrected.z() = (-imu_calib_info.gyro.g_xy_flu) * gyro_wo_b.x() +
                         imu_calib_info.gyro.g_yx_flu * gyro_wo_b.y() +
                         gyro_wo_b.z();
    ftf::EigenMapCovariance3d cov{angular_velocity_cov.data()};
    ftf::EigenMapCovariance3d cov_corrected{
        corrected_angular_velocity_cov.data()};
    cov_corrected = gyro_transform * cov * gyro_transform.transpose();
  }

  void correct_accel_raw(Eigen::Vector3d const &accel_raw,
                         Eigen::Vector3d &accel_corrected) {
    Eigen::Vector3d const accel_wo_b{imu_calib_info.accel.s_flu.cwiseProduct(
        accel_raw - imu_calib_info.accel.b_flu)};
    accel_corrected.x() = accel_wo_b.x() +
                          (-imu_calib_info.accel.a_yz_flu) * accel_wo_b.y() +
                          imu_calib_info.accel.a_zy_flu * accel_wo_b.z();
    accel_corrected.y() =
        accel_wo_b.y() + (-imu_calib_info.accel.a_zx_flu) * accel_wo_b.z();
    accel_corrected.z() = accel_wo_b.z();
    ftf::EigenMapCovariance3d cov{linear_acceleration_cov.data()};
    ftf::EigenMapCovariance3d cov_corrected{
        corrected_linear_acceleration_cov.data()};
    cov_corrected = accel_transform * cov * accel_transform.transpose();
  }

  void correct_imu_data_raw(Eigen::Vector3d const &gyro_raw,
                            Eigen::Vector3d const &accel_raw,
                            Eigen::Vector3d &gyro_corrected,
                            Eigen::Vector3d &accel_corrected) {
    correct_gyro_raw(gyro_raw, gyro_corrected);
    correct_accel_raw(accel_raw, accel_corrected);
  }

  /**
   * @brief Fill and publish IMU data_raw message; store linear acceleration
   * for IMU data
   * @param header      Message frame_id and timestamp
   * @param gyro_flu    Orientation in the base_link Forward-Left-Up frame
   * @param accel_flu   Linear acceleration in the base_link Forward-Left-Up
   * frame
   * @param accel_frd   Linear acceleration in the aircraft
   * Forward-Right-Down frame
   */
  void publish_imu_data_raw(const std_msgs::msg::Header &header,
                            const Eigen::Vector3d &gyro_flu,
                            const Eigen::Vector3d &accel_flu,
                            const Eigen::Vector3d &accel_frd) {
    if (pub_data_raw) {
      auto imu_msg = sensor_msgs::msg::Imu();

      // Fill message header
      imu_msg.header = header;

      tf2::toMsg(gyro_flu, imu_msg.angular_velocity);
      tf2::toMsg(accel_flu, imu_msg.linear_acceleration);

      imu_msg.orientation_covariance = unk_orientation_cov;
      imu_msg.angular_velocity_covariance = angular_velocity_cov;
      imu_msg.linear_acceleration_covariance = linear_acceleration_cov;

      // Publish message [ENU frame]
      imu_raw_pub->publish(imu_msg);
    }

    auto corrected_imu_msg = sensor_msgs::msg::Imu();

    // Fill message header
    corrected_imu_msg.header = header;

    correct_imu_data_raw(gyro_flu, accel_flu, corrected_gyro_flu,
                         corrected_accel_flu);
    tf2::toMsg(corrected_gyro_flu, corrected_imu_msg.angular_velocity);
    tf2::toMsg(corrected_accel_flu, corrected_imu_msg.linear_acceleration);

    // Save readings
    linear_accel_vec_flu = accel_flu;
    linear_accel_vec_frd = accel_frd;
    received_linear_accel = true;

    corrected_imu_msg.orientation_covariance = unk_orientation_cov;
    corrected_imu_msg.angular_velocity_covariance = angular_velocity_cov;
    corrected_imu_msg.linear_acceleration_covariance = linear_acceleration_cov;

    // Publish message [ENU frame]
    imu_corrected_pub->publish(corrected_imu_msg);
  }

  /**
   * @brief Publish magnetic field data
   * @param header	Message frame_id and timestamp
   * @param mag_field	Magnetic field in the base_link ENU frame
   */
  void publish_mag(const std_msgs::msg::Header &header,
                   const Eigen::Vector3d &mag_field) {
    auto magn_msg = sensor_msgs::msg::MagneticField();

    // Fill message header
    magn_msg.header = header;

    tf2::toMsg(mag_field, magn_msg.magnetic_field);
    magn_msg.magnetic_field_covariance = magnetic_cov;

    // Publish message [ENU frame]
    magn_pub->publish(magn_msg);
  }

  /* -*- message handlers -*- */

  /**
   * @brief Handle ATTITUDE MAVlink message.
   * Message specification: https://mavlink.io/en/messages/common.html#ATTITUDE
   * @param msg	Received Mavlink msg
   * @param att	ATTITUDE msg
   */
  void handle_attitude(const mavlink::mavlink_message_t *msg [[maybe_unused]],
                       mavlink::common::msg::ATTITUDE &att,
                       plugin::filter::SystemAndOk filter [[maybe_unused]]) {
    if (has_att_quat) {
      return;
    }

    /** Orientation on the NED-aicraft frame:
     *  @snippet src/plugins/imu.cpp ned_aircraft_orient1
     */
    // [ned_aircraft_orient1]
    auto ned_aircraft_orientation =
        ftf::quaternion_from_rpy(att.roll, att.pitch, att.yaw);
    // [ned_aircraft_orient1]

    /** Angular velocity on the NED-aicraft frame:
     *  @snippet src/plugins/imu.cpp ned_ang_vel1
     */
    // [frd_ang_vel1]
    auto gyro_frd =
        Eigen::Vector3d(att.rollspeed, att.pitchspeed, att.yawspeed);
    // [frd_ang_vel1]

    /** The RPY describes the rotation: aircraft->NED.
     *  It is required to change this to aircraft->base_link:
     *  @snippet src/plugins/imu.cpp ned->baselink->enu
     */
    // [ned->baselink->enu]
    auto enu_baselink_orientation =
        ftf::transform_orientation_aircraft_baselink(
            ftf::transform_orientation_ned_enu(ned_aircraft_orientation));
    // [ned->baselink->enu]

    /** The angular velocity expressed in the aircraft frame.
     *  It is required to apply the static rotation to get it into the base_link
     * frame:
     *  @snippet src/plugins/imu.cpp rotate_gyro
     */
    // [rotate_gyro]
    auto gyro_flu = ftf::transform_frame_aircraft_baselink(gyro_frd);
    // [rotate_gyro]

    publish_imu_data(att.time_boot_ms, enu_baselink_orientation,
                     ned_aircraft_orientation, gyro_flu, gyro_frd);
  }

  /**
   * @brief Handle ATTITUDE_QUATERNION MAVlink message.
   * Message specification:
   * https://mavlink.io/en/messages/common.html/#ATTITUDE_QUATERNION
   * @param msg		Received Mavlink msg
   * @param att_q		ATTITUDE_QUATERNION msg
   */
  void handle_attitude_quaternion(
      const mavlink::mavlink_message_t *msg [[maybe_unused]],
      mavlink::common::msg::ATTITUDE_QUATERNION &att_q,
      plugin::filter::SystemAndOk filter [[maybe_unused]]) {
    RCLCPP_INFO_EXPRESSION(get_logger(), !has_att_quat.exchange(true),
                           "IMU: Attitude quaternion IMU detected!");

    /** Orientation on the NED-aicraft frame:
     *  @snippet src/plugins/imu.cpp ned_aircraft_orient2
     */
    // [ned_aircraft_orient2]
    auto ned_aircraft_orientation =
        Eigen::Quaterniond(att_q.q1, att_q.q2, att_q.q3, att_q.q4);
    // [ned_aircraft_orient2]

    /** Angular velocity on the NED-aicraft frame:
     *  @snippet src/plugins/imu.cpp ned_ang_vel2
     */
    // [frd_ang_vel2]
    auto gyro_frd =
        Eigen::Vector3d(att_q.rollspeed, att_q.pitchspeed, att_q.yawspeed);
    // [frd_ang_vel2]

    /** MAVLink quaternion exactly matches Eigen convention.
     *  The RPY describes the rotation: aircraft->NED.
     *  It is required to change this to aircraft->base_link:
     *  @snippet src/plugins/imu.cpp ned->baselink->enu
     */
    auto enu_baselink_orientation =
        ftf::transform_orientation_aircraft_baselink(
            ftf::transform_orientation_ned_enu(ned_aircraft_orientation));

    /** The angular velocity expressed in the aircraft frame.
     *  It is required to apply the static rotation to get it into the base_link
     * frame:
     *  @snippet src/plugins/imu.cpp rotate_gyro
     */
    auto gyro_flu = ftf::transform_frame_aircraft_baselink(gyro_frd);

    publish_imu_data(att_q.time_boot_ms, enu_baselink_orientation,
                     ned_aircraft_orientation, gyro_flu, gyro_frd);
  }

  /**
   * @brief Handle HIGHRES_IMU MAVlink message.
   * Message specification:
   * https://mavlink.io/en/messages/common.html/#HIGHRES_IMU
   * @param msg		Received Mavlink msg
   * @param imu_hr	HIGHRES_IMU msg
   */
  void handle_highres_imu(const mavlink::mavlink_message_t *msg
                          [[maybe_unused]],
                          mavlink::common::msg::HIGHRES_IMU &imu_hr,
                          plugin::filter::SystemAndOk filter [[maybe_unused]]) {
    RCLCPP_INFO_EXPRESSION(get_logger(), !has_hr_imu.exchange(true),
                           "IMU: High resolution IMU detected!");

    auto header = uas->synchronized_header(frame_id, imu_hr.time_usec);
    /** @todo Make more paranoic check of HIGHRES_IMU.fields_updated
     */

    /** Check if accelerometer + gyroscope data are available.
     *  Data is expressed in aircraft frame it is required to rotate to the
     * base_link frame:
     *  @snippet src/plugins/imu.cpp accel_available
     */
    // [accel_available]
    if (imu_hr.fields_updated & ((7 << 3) | (7 << 0))) {
      auto gyro_flu = ftf::transform_frame_aircraft_baselink(
          Eigen::Vector3d(imu_hr.xgyro, imu_hr.ygyro, imu_hr.zgyro));

      auto accel_frd = Eigen::Vector3d(imu_hr.xacc, imu_hr.yacc, imu_hr.zacc);
      auto accel_flu = ftf::transform_frame_aircraft_baselink(accel_frd);

      publish_imu_data_raw(header, gyro_flu, accel_flu, accel_frd);
    }
    // [accel_available]

    /** Check if magnetometer data is available:
     *  @snippet src/plugins/imu.cpp mag_available
     */
    // [mag_available]
    if (imu_hr.fields_updated & (7 << 6)) {
      auto mag_field = ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(
          Eigen::Vector3d(imu_hr.xmag, imu_hr.ymag, imu_hr.zmag) *
          GAUSS_TO_TESLA);

      publish_mag(header, mag_field);
    }
    // [mag_available]

    /** Check if static pressure sensor data is available:
     *  @snippet src/plugins/imu.cpp static_pressure_available
     */
    // [static_pressure_available]
    if (imu_hr.fields_updated & (1 << 9)) {
      auto static_pressure_msg = sensor_msgs::msg::FluidPressure();

      static_pressure_msg.header = header;
      static_pressure_msg.fluid_pressure = imu_hr.abs_pressure;

      static_press_pub->publish(static_pressure_msg);
    }
    // [static_pressure_available]

    /** Check if differential pressure sensor data is available:
     *  @snippet src/plugins/imu.cpp differential_pressure_available
     */
    // [differential_pressure_available]
    if (imu_hr.fields_updated & (1 << 10)) {
      auto differential_pressure_msg = sensor_msgs::msg::FluidPressure();

      differential_pressure_msg.header = header;
      differential_pressure_msg.fluid_pressure = imu_hr.diff_pressure;

      diff_press_pub->publish(differential_pressure_msg);
    }
    // [differential_pressure_available]

    /** Check if temperature data is available:
     *  @snippet src/plugins/imu.cpp temperature_available
     */
    // [temperature_available]
    if (imu_hr.fields_updated & (1 << 12)) {
      auto temp_msg = sensor_msgs::msg::Temperature();

      temp_msg.header = header;
      temp_msg.temperature = imu_hr.temperature;

      temp_imu_pub->publish(temp_msg);
    }
    // [temperature_available]
  }

  /**
   * @brief Handle RAW_IMU MAVlink message.
   * Message specification: https://mavlink.io/en/messages/common.html/#RAW_IMU
   * @param msg		Received Mavlink msg
   * @param imu_raw	RAW_IMU msg
   */
  void handle_raw_imu(const mavlink::mavlink_message_t *msg [[maybe_unused]],
                      mavlink::common::msg::RAW_IMU &imu_raw,
                      plugin::filter::SystemAndOk filter [[maybe_unused]]) {
    RCLCPP_INFO_EXPRESSION(get_logger(), !has_raw_imu.exchange(true),
                           "IMU: Raw IMU message used.");

    if (has_hr_imu || has_scaled_imu) {
      return;
    }

    auto imu_msg = sensor_msgs::msg::Imu();
    auto header = uas->synchronized_header(frame_id, imu_raw.time_usec);

    /** @note APM send SCALED_IMU data as RAW_IMU
     */
    auto gyro_flu = ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(
        Eigen::Vector3d(imu_raw.xgyro, imu_raw.ygyro, imu_raw.zgyro) *
        MILLIRS_TO_RADSEC);
    auto accel_frd = Eigen::Vector3d(imu_raw.xacc, imu_raw.yacc, imu_raw.zacc);
    auto accel_flu =
        ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(accel_frd);

    if (uas->is_ardupilotmega()) {
      accel_frd *= MILLIG_TO_MS2;
      accel_flu *= MILLIG_TO_MS2;
    } else if (uas->is_px4()) {
      accel_frd *= MILLIMS2_TO_MS2;
      accel_flu *= MILLIMS2_TO_MS2;
    }

    publish_imu_data_raw(header, gyro_flu, accel_flu, accel_frd);

    if (!uas->is_ardupilotmega()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 60000,
          "IMU: linear acceleration on RAW_IMU known on APM only.");
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 60000,
          "IMU: ~imu/data_raw stores unscaled raw acceleration report.");
      linear_accel_vec_flu.setZero();
      linear_accel_vec_frd.setZero();
    }

    /** Magnetic field data:
     *  @snippet src/plugins/imu.cpp mag_field
     */
    // [mag_field]
    auto mag_field = ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(
        Eigen::Vector3d(imu_raw.xmag, imu_raw.ymag, imu_raw.zmag) *
        MILLIT_TO_TESLA);
    // [mag_field]

    publish_mag(header, mag_field);
  }

  /**
   * @brief Handle SCALED_IMU MAVlink message.
   * Message specification:
   * https://mavlink.io/en/messages/common.html/#SCALED_IMU
   * @param msg		Received Mavlink msg
   * @param imu_raw	SCALED_IMU msg
   */
  void handle_scaled_imu(const mavlink::mavlink_message_t *msg [[maybe_unused]],
                         mavlink::common::msg::SCALED_IMU &imu_raw,
                         plugin::filter::SystemAndOk filter [[maybe_unused]]) {
    if (has_hr_imu) {
      return;
    }

    RCLCPP_INFO_EXPRESSION(get_logger(), !has_scaled_imu.exchange(true),
                           "IMU: Scaled IMU message used.");

    auto header = uas->synchronized_header(frame_id, imu_raw.time_boot_ms);

    auto gyro_flu = ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(
        Eigen::Vector3d(imu_raw.xgyro, imu_raw.ygyro, imu_raw.zgyro) *
        MILLIRS_TO_RADSEC);
    auto accel_frd = Eigen::Vector3d(
        Eigen::Vector3d(imu_raw.xacc, imu_raw.yacc, imu_raw.zacc) *
        MILLIG_TO_MS2);
    auto accel_flu =
        ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(accel_frd);

    publish_imu_data_raw(header, gyro_flu, accel_flu, accel_frd);

    /** Magnetic field data:
     *  @snippet src/plugins/imu.cpp mag_field
     */
    auto mag_field = ftf::transform_frame_aircraft_baselink<Eigen::Vector3d>(
        Eigen::Vector3d(imu_raw.xmag, imu_raw.ymag, imu_raw.zmag) *
        MILLIT_TO_TESLA);

    publish_mag(header, mag_field);
  }

  /**
   * @brief Handle SCALED_PRESSURE MAVlink message.
   * Message specification:
   * https://mavlink.io/en/messages/common.html/#SCALED_PRESSURE
   * @param msg		Received Mavlink msg
   * @param press		SCALED_PRESSURE msg
   */
  void handle_scaled_pressure(const mavlink::mavlink_message_t *msg
                              [[maybe_unused]],
                              mavlink::common::msg::SCALED_PRESSURE &press,
                              plugin::filter::SystemAndOk filter
                              [[maybe_unused]]) {
    if (has_hr_imu) {
      return;
    }

    auto header = uas->synchronized_header(frame_id, press.time_boot_ms);

    auto temp_msg = sensor_msgs::msg::Temperature();
    temp_msg.header = header;
    temp_msg.temperature = press.temperature / 100.0;
    temp_baro_pub->publish(temp_msg);

    auto static_pressure_msg = sensor_msgs::msg::FluidPressure();
    static_pressure_msg.header = header;
    static_pressure_msg.fluid_pressure = press.press_abs * 100.0;
    static_press_pub->publish(static_pressure_msg);

    auto differential_pressure_msg = sensor_msgs::msg::FluidPressure();
    differential_pressure_msg.header = header;
    differential_pressure_msg.fluid_pressure = press.press_diff * 100.0;
    diff_press_pub->publish(differential_pressure_msg);
  }

  // Checks for connection and overrides variable values
  void connection_cb([[maybe_unused]] bool connected) override {
    has_hr_imu = false;
    has_raw_imu = false;
    has_scaled_imu = false;
    has_att_quat = false;
  }
};

} // namespace std_plugins
} // namespace mavros

#include <mavros/mavros_plugin_register_macro.hpp> // NOLINT
MAVROS_PLUGIN_REGISTER(mavros::std_plugins::IMUPlugin)
