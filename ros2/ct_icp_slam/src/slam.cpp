#include <iostream>

#include <glog/logging.h>

#include "nav_msgs/msg/odometry.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/convert.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

#include "dataset.hpp"
#include "evaluate_slam.hpp"
#include "io.hpp"
#include "odometry.hpp"
#include "utils.hpp"

#include "lgmath.hpp"

#define PCL_NO_PRECOMPILE
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ct_icp {

#define PCL_ADD_FLEXIBLE     \
  union EIGEN_ALIGN16 {      \
    __uint128_t raw_flex1;   \
    float data_flex1[4];     \
    struct {                 \
      float flex11;          \
      float flex12;          \
      float flex13;          \
      float flex14;          \
    };                       \
    struct {                 \
      float alpha_timestamp; \
      float timestamp;       \
      float radial_velocity; \
    };                       \
  };

struct EIGEN_ALIGN16 _PCLPoint3D {
  PCL_ADD_POINT4D;
  PCL_ADD_FLEXIBLE;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

struct PCLPoint3D : public _PCLPoint3D {
  inline PCLPoint3D() {
    x = y = z = 0.0f;
    data[3] = 1.0f;
    raw_flex1 = 0;
  }

  inline PCLPoint3D(const _PCLPoint3D &p) {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
    raw_flex1 = p.raw_flex1;
  }

  inline PCLPoint3D(const Point3D &p) {
    x = (float)p.pt[0];
    y = (float)p.pt[1];
    z = (float)p.pt[2];
    data[3] = 1.0f;
    alpha_timestamp = p.alpha_timestamp;
    timestamp = p.timestamp;
    radial_velocity = p.radial_velocity;
  }

  inline PCLPoint3D(const Eigen::Vector3d &p) {
    x = (float)p[0];
    y = (float)p[1];
    z = (float)p[2];
    data[3] = 1.0f;
  }
};

enum SLAM_VIZ_MODE {
  AGGREGATED,  // Will display all aggregated frames
  KEYPOINTS    // Will display at each step the keypoints used
};

// Parameters to run the SLAM
struct SLAMOptions {
  DatasetOptions dataset_options;

  OdometryOptions odometry_options;

  int max_num_threads = 1;               // The maximum number of threads running in parallel the Dataset acquisition
  bool save_trajectory = true;           // whether to save the trajectory
  bool suspend_on_failure = false;       // Whether to suspend the execution once an error is detected
  std::string output_dir = "./outputs";  // The output path (relative or absolute) to save the pointclouds
  bool all_sequences = true;             // Whether to run the algorithm on all sequences of the dataset found on disk
  std::string sequence;                  // The desired sequence (only applicable if `all_sequences` is false)
  int start_index = 0;      // The start index of the sequence (only applicable if `all_sequences` is false)
  int max_frames = -1;      // The maximum number of frames to register (if -1 all frames in the Dataset are registered)
  bool with_viz3d = false;  // Whether to display timing and debug information
  SLAM_VIZ_MODE viz_mode = AGGREGATED;  // The visualization mode for the point clouds (in AGGREGATED, KEYPOINTS)

  struct {
    bool odometry = true;
    bool raw_points = true;
    bool sampled_points = true;
    bool map_points = true;
    Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
  } visualization_options;
};

#define ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type) \
  receiver = node->declare_parameter<type>(prefix + #param, receiver);
#define ROS2_PARAM(node, receiver, prefix, param, type)   \
  ROS2_PARAM_NO_LOG(node, receiver, prefix, param, type); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << receiver << std::endl;
#define ROS2_PARAM_CLAUSE(node, config, prefix, param, type)                   \
  config.param = node->declare_parameter<type>(prefix + #param, config.param); \
  LOG(WARNING) << "Parameter " << prefix + #param << " = " << config.param << std::endl;

ct_icp::SLAMOptions load_options(const rclcpp::Node::SharedPtr &node) {
  ct_icp::SLAMOptions options;
  std::string prefix;

  /// slam options
  {
    ROS2_PARAM_CLAUSE(node, options, prefix, max_num_threads, int);
    ROS2_PARAM_CLAUSE(node, options, prefix, save_trajectory, bool);
    ROS2_PARAM_CLAUSE(node, options, prefix, suspend_on_failure, bool);

    ROS2_PARAM_CLAUSE(node, options, prefix, output_dir, std::string);
    if (!options.output_dir.empty() && options.output_dir[options.output_dir.size() - 1] != '/')
      options.output_dir += '/';

    ROS2_PARAM_CLAUSE(node, options, prefix, all_sequences, bool);
    ROS2_PARAM_CLAUSE(node, options, prefix, sequence, std::string);
    ROS2_PARAM_CLAUSE(node, options, prefix, start_index, int);
    ROS2_PARAM_CLAUSE(node, options, prefix, max_frames, int);
    ROS2_PARAM_CLAUSE(node, options, prefix, with_viz3d, bool);

    std::string viz_mode;
    ROS2_PARAM(node, viz_mode, prefix, viz_mode, std::string);
    if (viz_mode == "AGGREGATED") options.viz_mode = AGGREGATED;
    if (viz_mode == "KEYPOINTS") options.viz_mode = KEYPOINTS;
  }

  /// visualization options
  {
    auto &visualization_options = options.visualization_options;
    prefix = "visualization_options.";

    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, odometry, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, raw_points, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, sampled_points, bool);
    ROS2_PARAM_CLAUSE(node, visualization_options, prefix, map_points, bool);

    std::vector<double> T_sr_vec;
    ROS2_PARAM_NO_LOG(node, T_sr_vec, prefix, T_sr_vec, std::vector<double>);
    if ((T_sr_vec.size() != 6) && (T_sr_vec.size() != 0))
      throw std::invalid_argument{"T_sr malformed. Must be 6 elements!"};
    if (T_sr_vec.size() == 6)
      visualization_options.T_sr = lgmath::se3::vec2tran(Eigen::Matrix<double, 6, 1>(T_sr_vec.data()));
    LOG(WARNING) << "Parameter " << prefix + "T_sr"
                 << " = " << std::endl
                 << visualization_options.T_sr << std::endl;
  }

  /// dataset options
  {
    auto &dataset_options = options.dataset_options;
    prefix = "dataset_options.";

    std::string dataset;
    ROS2_PARAM(node, dataset, prefix, dataset, std::string);
    if (dataset == "KITTI_raw")
      dataset_options.dataset = KITTI_raw;
    else if (dataset == "KITTI_CARLA")
      dataset_options.dataset = KITTI_CARLA;
    else if (dataset == "KITTI")
      dataset_options.dataset = KITTI;
    else if (dataset == "KITTI-360")
      dataset_options.dataset = KITTI_360;
    else if (dataset == "NCLT")
      dataset_options.dataset = NCLT;
    else if (dataset == "BOREAS")
      dataset_options.dataset = BOREAS;
    else if (dataset == "AEVA")
      dataset_options.dataset = AEVA;
    else if (dataset == "PLY_DIRECTORY")
      dataset_options.dataset = PLY_DIRECTORY;
    else
      throw std::runtime_error("Invalid dataset");

    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, root_path, std::string);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, fail_if_incomplete, bool);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, min_dist_lidar_center, float);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, max_dist_lidar_center, float);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, nclt_num_aggregated_pc, int);
  }

  /// odometry options
  {
    auto &odometry_options = options.odometry_options;
    prefix = "odometry_options.";

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_sample_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_num_frames, int);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, sample_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_distance, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_num_points_in_voxel, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, min_distance_points, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, distance_error_threshold, double);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_minimal_level, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_registration, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_full_voxel_threshold, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_empty_voxel_threshold, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_neighborhood_min_dist, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_neighborhood_min_orientation, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_relative_trans_threshold, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_fail_early, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_num_attempts, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_num_attempts_when_rotation, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_max_voxel_neighborhood, short);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_threshold_ego_orientation, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_threshold_relative_orientation, double)

    std::string motion_compensation;
    ROS2_PARAM(node, motion_compensation, prefix, motion_compensation, std::string);
    if (motion_compensation == "NONE")
      odometry_options.motion_compensation = ct_icp::NONE;
    else if (motion_compensation == "CONSTANT_VELOCITY")
      odometry_options.motion_compensation = ct_icp::CONSTANT_VELOCITY;
    else if (motion_compensation == "ITERATIVE")
      odometry_options.motion_compensation = ct_icp::ITERATIVE;
    else if (motion_compensation == "CONTINUOUS")
      odometry_options.motion_compensation = ct_icp::CONTINUOUS;
    else {
      LOG(WARNING) << "Parameter " << prefix + "motion_compensation"
                   << " not specified. Using default value: "
                   << "CONTINUOUS";
    }

    std::string initialization;
    ROS2_PARAM(node, initialization, prefix, initialization, std::string);
    if (initialization == "INIT_NONE")
      odometry_options.initialization = ct_icp::INIT_NONE;
    else if (initialization == "INIT_CONSTANT_VELOCITY")
      odometry_options.initialization = ct_icp::INIT_CONSTANT_VELOCITY;
    else {
      LOG(WARNING) << "Parameter " << prefix + "initialization"
                   << " not specified. Using default value: "
                   << "INIT_CONSTANT_VELOCITY";
    }

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_print, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_viz, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, log_to_file, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, log_file_destination, std::string);
  }

  /// ct_icp options
  {
    auto &ct_icp_options = options.odometry_options.ct_icp_options;
    prefix = "odometry_options.ct_icp_options.";

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_voxel_occupancy, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, init_num_frames, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, size_voxel_map, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, num_iters_icp, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, min_number_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, voxel_neighborhood, short);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, power_planarity, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, estimate_normal_from_neighborhood, bool);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_number_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_dist_to_plane_ct_icp, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_orientation_norm, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_translation_norm, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, point_to_plane_with_distortion, bool);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_num_residuals, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, min_num_residuals, int);

    std::string distance;
    ROS2_PARAM(node, distance, prefix, distance, std::string);
    if (distance == "POINT_TO_PLANE")
      ct_icp_options.distance = POINT_TO_PLANE;
    else if (distance == "CT_POINT_TO_PLANE")
      ct_icp_options.distance = CT_POINT_TO_PLANE;
    else {
      LOG(WARNING) << "Parameter " << prefix + "distance"
                   << " not specified. Using default value: "
                   << "CT_POINT_TO_PLANE";
    }

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, num_closest_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_location_consistency, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_constant_velocity, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_small_velocity, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_orientation_consistency, double);

    std::string weighting_scheme;
    ROS2_PARAM(node, weighting_scheme, prefix, weighting_scheme, std::string);
    if (weighting_scheme == "PLANARITY")
      ct_icp_options.weighting_scheme = PLANARITY;
    else if (weighting_scheme == "NEIGHBORHOOD")
      ct_icp_options.weighting_scheme = NEIGHBORHOOD;
    else if (weighting_scheme == "ALL")
      ct_icp_options.weighting_scheme = ALL;
    else {
      LOG(WARNING) << "Parameter " << prefix + "weighting_scheme"
                   << " not specified. Using default value: "
                   << "ALL";
    }

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, weight_alpha, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, weight_neighborhood, double);

    std::string solver;
    ROS2_PARAM(node, solver, prefix, solver, std::string);
    if (solver == "GN")
      ct_icp_options.solver = GN;
    else if (solver == "CERES")
      ct_icp_options.solver = CERES;
    else if (solver == "STEAM")
      ct_icp_options.solver = STEAM;
    else {
      LOG(WARNING) << "Parameter " << prefix + "solver"
                   << " not specified. Using default value: "
                   << "GN";
    }

    std::string loss_function;
    ROS2_PARAM(node, loss_function, prefix, loss_function, std::string);
    if (loss_function == "STANDARD")
      ct_icp_options.loss_function = STANDARD;
    else if (loss_function == "CAUCHY")
      ct_icp_options.loss_function = CAUCHY;
    else if (loss_function == "HUBER")
      ct_icp_options.loss_function = HUBER;
    else if (loss_function == "TOLERANT")
      ct_icp_options.loss_function = TOLERANT;
    else if (loss_function == "TRUNCATED")
      ct_icp_options.loss_function = TRUNCATED;
    else {
      LOG(WARNING) << "Parameter " << prefix + "loss_function"
                   << " not specified. Using default value: "
                   << "CAUCHY";
    }

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_max_num_iters, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_num_threads, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_sigma, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_tolerant_min_threshold, double);

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, debug_print, bool);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, debug_viz, bool);

    std::string viz_mode;
    ROS2_PARAM(node, viz_mode, prefix, viz_mode, std::string);
    if (viz_mode == "NORMAL")
      ct_icp_options.viz_mode = NORMAL;
    else if (viz_mode == "WEIGHT")
      ct_icp_options.viz_mode = WEIGHT;
    else if (viz_mode == "TIMESTAMP")
      ct_icp_options.viz_mode = TIMESTAMP;
    else {
      LOG(WARNING) << "Parameter " << prefix + "viz_mode"
                   << " not specified. Using default value: "
                   << "TIMESTAMP";
    }

    {
      auto &steam = options.odometry_options.ct_icp_options.steam;
      prefix = "odometry_options.ct_icp_options.steam.";

      std::vector<double> T_sr_vec;
      ROS2_PARAM_NO_LOG(node, T_sr_vec, prefix, T_sr_vec, std::vector<double>);
      if ((T_sr_vec.size() != 6) && (T_sr_vec.size() != 0))
        throw std::invalid_argument{"T_sr malformed. Must be 6 elements!"};
      if (T_sr_vec.size() == 6) steam.T_sr = lgmath::se3::vec2tran(Eigen::Matrix<double, 6, 1>(T_sr_vec.data()));
      LOG(WARNING) << "Parameter " << prefix + "T_sr"
                   << " = " << std::endl
                   << steam.T_sr << std::endl;

      std::vector<double> qc_inv_diag;
      ROS2_PARAM_NO_LOG(node, qc_inv_diag, prefix, qc_inv_diag, std::vector<double>);
      if ((qc_inv_diag.size() != 6) && (qc_inv_diag.size() != 0))
        throw std::invalid_argument{"Qc diagonal malformed. Must be 6 elements!"};
      if (qc_inv_diag.size() == 6)
        steam.qc_inv.diagonal() << qc_inv_diag[0], qc_inv_diag[1], qc_inv_diag[2], qc_inv_diag[3], qc_inv_diag[4],
            qc_inv_diag[5];
      LOG(WARNING) << "Parameter " << prefix + "qc_inv_diag"
                   << " = " << steam.qc_inv.diagonal().transpose() << std::endl;

      ROS2_PARAM_CLAUSE(node, steam, prefix, num_extra_states, int);
      ROS2_PARAM_CLAUSE(node, steam, prefix, add_prev_state, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, num_extra_prev_states, int);
      ROS2_PARAM_CLAUSE(node, steam, prefix, lock_prev_pose, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, lock_prev_vel, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, prev_pose_as_prior, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, prev_vel_as_prior, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, no_prev_state_iters, int);
      ROS2_PARAM_CLAUSE(node, steam, prefix, association_after_adding_prev_state, bool);

      ROS2_PARAM_CLAUSE(node, steam, prefix, use_vp, bool);

      std::vector<double> vp_cov_diag;
      ROS2_PARAM_NO_LOG(node, vp_cov_diag, prefix, vp_cov_diag, std::vector<double>);
      if ((vp_cov_diag.size() != 6) && (vp_cov_diag.size() != 0))
        throw std::invalid_argument{"Velocity prior cov malformed. Must be 6 elements!"};
      if (vp_cov_diag.size() == 6)
        steam.vp_cov.diagonal() << vp_cov_diag[0], vp_cov_diag[1], vp_cov_diag[2], vp_cov_diag[3], vp_cov_diag[4],
            vp_cov_diag[5];
      LOG(WARNING) << "Parameter " << prefix + "vp_cov_diag"
                   << " = " << steam.vp_cov.diagonal().transpose() << std::endl;

      ROS2_PARAM_CLAUSE(node, steam, prefix, p2p_initial_iters, int);
      ROS2_PARAM_CLAUSE(node, steam, prefix, p2p_initial_max_dist, double);
      ROS2_PARAM_CLAUSE(node, steam, prefix, p2p_refined_max_dist, double);
      std::string p2p_loss_func;
      ROS2_PARAM(node, p2p_loss_func, prefix, p2p_loss_func, std::string);
      if (p2p_loss_func == "L2")
        steam.p2p_loss_func = STEAM_LOSS_FUNC::L2;
      else if (p2p_loss_func == "DCS")
        steam.p2p_loss_func = STEAM_LOSS_FUNC::DCS;
      else if (p2p_loss_func == "CAUCHY")
        steam.p2p_loss_func = STEAM_LOSS_FUNC::CAUCHY;
      else if (p2p_loss_func == "GM")
        steam.p2p_loss_func = STEAM_LOSS_FUNC::GM;
      else {
        LOG(WARNING) << "Parameter " << prefix + "p2p_loss_func"
                     << " not specified. Using default value: "
                     << "L2";
      }
      ROS2_PARAM_CLAUSE(node, steam, prefix, p2p_loss_sigma, double);

      ROS2_PARAM_CLAUSE(node, steam, prefix, use_rv, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, merge_p2p_rv, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, rv_cov_inv, double);
      ROS2_PARAM_CLAUSE(node, steam, prefix, rv_loss_threshold, double);

      ROS2_PARAM_CLAUSE(node, steam, prefix, verbose, bool);
      ROS2_PARAM_CLAUSE(node, steam, prefix, max_iterations, int);
      ROS2_PARAM_CLAUSE(node, steam, prefix, num_threads, int);
    }
  }

  return options;
}

}  // namespace ct_icp

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(
    ct_icp::PCLPoint3D,
    // cartesian coordinates
    (float, x, x)
    (float, y, y)
    (float, z, z)
    // random stuff
    (float, flex11, flex11)
    (float, flex12, flex12)
    (float, flex13, flex13)
    (float, flex14, flex14))
// clang-format on

int main(int argc, char **argv) {
  using namespace ct_icp;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ct_icp_slam");
  auto odometry_publisher = node->create_publisher<nav_msgs::msg::Odometry>("/ct_icp_odometry", 10);
  auto tf_static_bc = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
  auto tf_bc = std::make_shared<tf2_ros::TransformBroadcaster>(node);
  auto raw_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/ct_icp_raw", 2);
  auto sampled_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/ct_icp_sampled", 2);
  auto map_points_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("/ct_icp_map", 2);

  auto to_pc2_msg = [](const auto &points, const std::string &frame_id = "map") {
    pcl::PointCloud<PCLPoint3D> points_pcl;
    points_pcl.reserve(points.size());
    for (auto &pt : points) points_pcl.emplace_back(pt);
    sensor_msgs::msg::PointCloud2 points_msg;
    pcl::toROSMsg(points_pcl, points_msg);
    points_msg.header.frame_id = frame_id;
    // points_msg.header.stamp = rclcpp::Time(stamp);
    return points_msg;
  };

  // Logging
  FLAGS_log_dir = node->declare_parameter<std::string>("log_dir", "/tmp");
  FLAGS_alsologtostderr = 1;
  fs::create_directories(FLAGS_log_dir);
  google::InitGoogleLogging(argv[0]);
  LOG(WARNING) << "Logging to " << FLAGS_log_dir;

  // Read parameters
  const auto options = ct_icp::load_options(node);

  // Publish sensor vehicle transformations
  auto T_rs_msg = tf2::eigenToTransform(Eigen::Affine3d(options.visualization_options.T_sr.inverse()));
  T_rs_msg.header.frame_id = "vehicle";
  T_rs_msg.child_frame_id = "lidar";
  tf_static_bc->sendTransform(T_rs_msg);

  // Build the Output_dir
  if (!fs::exists(options.dataset_options.root_path))
    LOG(WARNING) << "The directory " << options.dataset_options.root_path << " does not exist";
  LOG(WARNING) << "Creating directory " << options.output_dir << std::endl;
  fs::create_directories(options.output_dir);

  // Get the dataset
  auto sequences = ct_icp::get_sequences(options.dataset_options);
  if (!options.all_sequences) {
    // Select a specific sequence
    int seq_idx = -1;
    for (int idx(0); idx < (int)sequences.size(); ++idx) {
      auto &sequence = sequences[idx];
      if (sequence.sequence_name == options.sequence) {
        seq_idx = idx;
        break;
      }
    }

    if (seq_idx == -1) {
      LOG(ERROR) << "Could not find the sequence " << options.sequence << ". Exiting." << std::endl;
      return 1;
    }

    auto selected_sequence = std::move(sequences[seq_idx]);
    sequences.resize(1);
    sequences[0] = std::move(selected_sequence);
  }
  int num_sequences = (int)sequences.size();

  //
  std::map<std::string, ct_icp::seq_errors> sequence_name_to_errors;
  bool dataset_with_gt = false;
  double all_seq_registration_elapsed_ms = 0.0;
  int all_seq_num_frames = 0;
  double average_rpe_on_seq = 0.0;
  int nb_seq_with_gt = 0;

  for (int i = 0; i < num_sequences; ++i) {  // num_sequences

    int sequence_id = sequences[i].sequence_id;
    ct_icp::Odometry ct_icp_odometry(&options.odometry_options);

    double registration_elapsed_ms = 0.0;

    auto iterator_ptr = get_dataset_sequence(options.dataset_options, sequence_id);

    double avg_number_of_attempts = 0.0;
    int frame_id(0);
    if (!options.all_sequences && options.start_index > 0) {
      LOG(WARNING) << "Starting at frame " << options.start_index << std::endl;
      iterator_ptr->SetInitFrame(options.start_index);
    }
    while (iterator_ptr->HasNext() && (options.max_frames < 0 || frame_id < options.max_frames)) {
      auto time_start_frame = std::chrono::steady_clock::now();
      std::vector<Point3D> frame = iterator_ptr->Next();

      /// raw points
      if (options.visualization_options.raw_points) {
        auto &raw_points = frame;
        auto raw_points_msg = to_pc2_msg(raw_points, "lidar");
        raw_points_publisher->publish(raw_points_msg);
      }

      auto time_read_pointcloud = std::chrono::steady_clock::now();

      auto summary = ct_icp_odometry.RegisterFrame(frame);
      avg_number_of_attempts += summary.number_of_attempts;
      auto time_register_frame = std::chrono::steady_clock::now();

      std::chrono::duration<double> total_elapsed = time_register_frame - time_start_frame;
      std::chrono::duration<double> registration_elapsed = time_register_frame - time_read_pointcloud;

      registration_elapsed_ms += registration_elapsed.count() * 1000;
      all_seq_registration_elapsed_ms += registration_elapsed.count() * 1000;

      /// publish to rviz
      if (options.visualization_options.odometry) {
        Eigen::Matrix4d T_ws = Eigen::Matrix4d::Identity();
        T_ws.block<3, 3>(0, 0) = summary.frame.begin_R;
        T_ws.block<3, 1>(0, 3) = summary.frame.begin_t;
        Eigen::Matrix4d T_wr = T_ws * options.visualization_options.T_sr;

        /// odometry
        nav_msgs::msg::Odometry odometry;
        odometry.header.frame_id = "map";
        // odometry.header.stamp = rclcpp::Time(stamp);
        odometry.pose.pose = tf2::toMsg(Eigen::Affine3d(T_wr));
        odometry_publisher->publish(odometry);

        /// tf
        auto T_wr_msg = tf2::eigenToTransform(Eigen::Affine3d(T_wr));
        T_wr_msg.header.frame_id = "map";
        // T_wr_msg.header.stamp = rclcpp::Time(stamp);
        T_wr_msg.child_frame_id = "vehicle";
        tf_bc->sendTransform(T_wr_msg);
      }
      if (options.visualization_options.sampled_points) {
        /// sampled points
        auto &sampled_points = summary.corrected_points;
        auto sampled_points_msg = to_pc2_msg(sampled_points, "map");
        sampled_points_publisher->publish(sampled_points_msg);
      }
      if (options.visualization_options.map_points) {
        /// map points
        auto map_points = ct_icp_odometry.GetLocalMap();
        auto map_points_msg = to_pc2_msg(map_points, "map");
        map_points_publisher->publish(map_points_msg);
      }

      if (!rclcpp::ok()) {
        LOG(WARNING) << "Shutting down due to ctrl-c." << std::endl;
        return 0;
      }

      if (!summary.success) {
        LOG(ERROR) << "Error while running SLAM for sequence " << sequence_id << ", at frame index " << frame_id
                   << ". Error Message: " << summary.error_message << std::endl;
        if (options.suspend_on_failure) {
          return 1;
        }
        break;
      }
      frame_id++;
      all_seq_num_frames++;
    }

    avg_number_of_attempts /= (frame_id - 1);

    auto trajectory = ct_icp_odometry.Trajectory();
    auto trajectory_absolute_poses = transform_trajectory_frame(options.dataset_options, trajectory, sequence_id);
    // Save Trajectory And Compute metrics for trajectory with ground truths

    std::string _sequence_name = sequence_name(options.dataset_options, sequence_id);
    if (options.save_trajectory) {
      // Save trajectory to disk
      auto filepath = options.output_dir + _sequence_name + "_poses.txt";
      auto dual_poses_filepath = options.output_dir + _sequence_name + "_dual_poses.txt";
      if (!SavePoses(filepath, trajectory_absolute_poses) || !SaveTrajectoryFrame(dual_poses_filepath, trajectory)) {
        LOG(ERROR) << "Error while saving the poses to " << filepath << std::endl;
        LOG(ERROR) << "Make sure output directory " << options.output_dir << " exists" << std::endl;

        if (options.suspend_on_failure) {
          return 1;
        }
      }
    }

    // Evaluation
    if (has_ground_truth(options.dataset_options, sequence_id)) {
      dataset_with_gt = true;
      nb_seq_with_gt++;

      auto ground_truth_poses = load_ground_truth(options.dataset_options, sequence_id);

      bool valid_trajectory = ground_truth_poses.size() == trajectory_absolute_poses.size();
      if (!valid_trajectory) ground_truth_poses.resize(trajectory_absolute_poses.size());

      ct_icp::seq_errors seq_error = ct_icp::eval(ground_truth_poses, trajectory_absolute_poses);
      seq_error.average_elapsed_ms = registration_elapsed_ms / frame_id;
      seq_error.mean_num_attempts = avg_number_of_attempts;

      LOG(WARNING) << "[RESULTS] Sequence " << _sequence_name << std::endl;
      if (!valid_trajectory) {
        LOG(WARNING) << "Invalid Trajectory, Failed after " << ground_truth_poses.size() << std::endl;
        LOG(WARNING) << "Num Poses : " << seq_error.mean_rpe << std::endl;
      }
      LOG(WARNING) << "Average Number of Attempts : " << avg_number_of_attempts << std::endl;
      LOG(WARNING) << "Mean RPE : " << seq_error.mean_rpe << std::endl;
      LOG(WARNING) << "Mean RPE 2D : " << seq_error.mean_rpe_2d << std::endl;
      LOG(WARNING) << "Mean APE : " << seq_error.mean_ape << std::endl;
      LOG(WARNING) << "Max APE : " << seq_error.max_ape << std::endl;
      LOG(WARNING) << "Mean Local Error : " << seq_error.mean_local_err << std::endl;
      LOG(WARNING) << "Max Local Error : " << seq_error.max_local_err << std::endl;
      LOG(WARNING) << "Index Max Local Error : " << seq_error.index_max_local_err << std::endl;
      LOG(WARNING) << "Average Duration : " << registration_elapsed_ms / frame_id << std::endl;
      LOG(WARNING) << std::endl;

      average_rpe_on_seq += seq_error.mean_rpe;

      {
        sequence_name_to_errors[_sequence_name] = seq_error;
        // Save Metrics to file
#if false
        ct_icp::SaveMetrics(sequence_name_to_errors, options.output_dir + "metrics.yaml", valid_trajectory);
#endif
      };
    }
  }

  if (dataset_with_gt) {
    LOG(WARNING) << std::endl;
    double all_seq_rpe_t = 0.0;
    double all_seq_rpe_r = 0.0;
    double num_total_errors = 0.0;
    for (auto &pair : sequence_name_to_errors) {
      for (auto &tab_error : pair.second.tab_errors) {
        all_seq_rpe_t += tab_error.t_err;
        all_seq_rpe_r += tab_error.r_err;
        num_total_errors += 1;
      }
    }
    LOG(WARNING) << "KITTI metric translation/rotation : " << (all_seq_rpe_t / num_total_errors) * 100 << " "
                 << (all_seq_rpe_r / num_total_errors) * 180.0 / M_PI << std::endl;
    LOG(WARNING) << "Average RPE on seq : " << average_rpe_on_seq / nb_seq_with_gt;
  }

  LOG(WARNING) << std::endl;
  LOG(WARNING) << "Average registration time for all sequences (ms) : "
               << all_seq_registration_elapsed_ms / all_seq_num_frames << std::endl;

  // rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}