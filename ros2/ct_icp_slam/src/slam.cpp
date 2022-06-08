#include <iostream>

#include <glog/logging.h>

#include "rclcpp/rclcpp.hpp"

#include "dataset.hpp"
#include "evaluate_slam.hpp"
#include "io.hpp"
#include "odometry.hpp"
#include "utils.hpp"

namespace ct_icp {

enum SLAM_VIZ_MODE {
  AGGREGATED,  // Will display all aggregated frames
  KEYPOINTS    // Will display at each step the keypoints used
};

// Parameters to run the SLAM
struct SLAMOptions {
  DatasetOptions dataset_options;

  OdometryOptions odometry_options;

  int max_num_threads = 1;  // The maximum number of threads running in parallel the Dataset acquisition

  bool suspend_on_failure = false;  // Whether to suspend the execution once an error is detected

  bool save_trajectory = true;  // whether to save the trajectory

  std::string output_dir = "./outputs";  // The output path (relative or absolute) to save the pointclouds

  bool all_sequences = true;  // Whether to run the algorithm on all sequences of the dataset found on disk

  std::string sequence;  // The desired sequence (only applicable if `all_sequences` is false)

  int start_index = 0;  // The start index of the sequence (only applicable if `all_sequences` is false)

  int max_frames = -1;  // The maximum number of frames to register (if -1 all frames in the Dataset are registered)

  bool with_viz3d = true;  // Whether to display timing and debug information

  SLAM_VIZ_MODE viz_mode = KEYPOINTS;  // The visualization mode for the point clouds (in AGGREGATED, KEYPOINTS)
};

#define ROS2_PARAM(node, receiver, prefix, param, type)                \
  receiver = node->declare_parameter<type>(prefix + #param, receiver); \
  LOG(INFO) << "Parameter " << prefix + #param << " = " << receiver << std::endl;
#define ROS2_PARAM_CLAUSE(node, config, prefix, param, type)                   \
  config.param = node->declare_parameter<type>(prefix + #param, config.param); \
  LOG(INFO) << "Parameter " << prefix + #param << " = " << config.param << std::endl;

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

    ROS2_PARAM_CLAUSE(node, options, prefix, sequence, std::string);
    ROS2_PARAM_CLAUSE(node, options, prefix, start_index, int);
    ROS2_PARAM_CLAUSE(node, options, prefix, all_sequences, bool);
    ROS2_PARAM_CLAUSE(node, options, prefix, with_viz3d, bool);

    std::string viz_mode;
    ROS2_PARAM(node, viz_mode, prefix, viz_mode, std::string);
    if (viz_mode == "AGGREGATED") options.viz_mode = AGGREGATED;
    if (viz_mode == "KEYPOINTS") options.viz_mode = KEYPOINTS;
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
    else if (dataset == "PLY_DIRECTORY")
      dataset_options.dataset = PLY_DIRECTORY;
    else
      throw std::runtime_error("Invalid dataset");

    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, root_path, std::string);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, fail_if_incomplete, bool);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, min_dist_lidar_center, float);
    ROS2_PARAM_CLAUSE(node, dataset_options, prefix, max_dist_lidar_center, float);
  }

  /// odometry options
  {
    auto &odometry_options = options.odometry_options;
    prefix = "odometry_options.";

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, sample_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_distance, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, max_num_points_in_voxel, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_print, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, debug_viz, bool);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, min_distance_points, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, distance_error_threshold, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_num_frames, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, init_sample_voxel_size, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, log_to_file, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, log_file_destination, std::string);

    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_minimal_level, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_registration, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_full_voxel_threshold, double);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_fail_early, bool);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_num_attempts, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_max_voxel_neighborhood, int);
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_threshold_relative_orientation, double)
    ROS2_PARAM_CLAUSE(node, odometry_options, prefix, robust_threshold_ego_orientation, double);

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
    else
      throw std::runtime_error("Invalid motion_compensation");

    std::string initialization;
    ROS2_PARAM(node, initialization, prefix, initialization, std::string);
    if (initialization == "INIT_NONE")
      odometry_options.initialization = ct_icp::INIT_NONE;
    else if (initialization == "INIT_CONSTANT_VELOCITY")
      odometry_options.initialization = ct_icp::INIT_CONSTANT_VELOCITY;
    else
      throw std::runtime_error("Invalid initialization");
  }

  /// ct_icp options
  {
    auto &ct_icp_options = options.odometry_options.ct_icp_options;
    prefix = "odometry_options.ct_icp_options.";

    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_voxel_occupancy, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, size_voxel_map, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, num_iters_icp, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, min_number_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, voxel_neighborhood, short);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_number_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_dist_to_plane_ct_icp, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_orientation_norm, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, threshold_translation_norm, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, debug_print, bool);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, point_to_plane_with_distortion, bool);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, num_closest_neighbors, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_constant_velocity, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_location_consistency, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_small_velocity, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, beta_orientation_consistency, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_max_num_iters, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_num_threads, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_sigma, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, min_num_residuals, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, max_num_residuals, int);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, weight_alpha, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, weight_neighborhood, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, ls_tolerant_min_threshold, double);
    ROS2_PARAM_CLAUSE(node, ct_icp_options, prefix, debug_viz, bool);

    std::string distance;
    ROS2_PARAM(node, distance, prefix, distance, std::string);
    if (distance == "POINT_TO_PLANE")
      ct_icp_options.distance = POINT_TO_PLANE;
    else if (distance == "CT_POINT_TO_PLANE")
      ct_icp_options.distance = CT_POINT_TO_PLANE;
    else
      throw std::runtime_error("Invalid distance");

    std::string viz_mode;
    ROS2_PARAM(node, viz_mode, prefix, viz_mode, std::string);
    if (viz_mode == "NORMAL")
      ct_icp_options.viz_mode = NORMAL;
    else if (viz_mode == "WEIGHT")
      ct_icp_options.viz_mode = WEIGHT;
    else if (viz_mode == "TIMESTAMP")
      ct_icp_options.viz_mode = TIMESTAMP;
    else
      ct_icp_options.viz_mode = TIMESTAMP;  // default

    std::string solver;
    ROS2_PARAM(node, solver, prefix, solver, std::string);
    if (solver == "GN")
      ct_icp_options.solver = GN;
    else if (solver == "CERES")
      ct_icp_options.solver = CERES;
    else if (solver == "STEAM")
      ct_icp_options.solver = STEAM;
    else
      throw std::runtime_error("Invalid solver");

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
    else
      throw std::runtime_error("Invalid loss_function");
  }

  return options;
}

}  // namespace ct_icp

int main(int argc, char **argv) {
  using namespace ct_icp;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ct_icp_slam");

  // Logging
  FLAGS_log_dir = node->declare_parameter<std::string>("log_dir", "/tmp");
  FLAGS_alsologtostderr = 1;
  fs::create_directories(FLAGS_log_dir);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Logging to " << FLAGS_log_dir;

  // Read parameters
  const auto options = ct_icp::load_options(node);

  // Build the Output_dir
  if (!fs::exists(options.dataset_options.root_path))
    LOG(INFO) << "The directory " << options.dataset_options.root_path << " does not exist";
  LOG(INFO) << "Creating directory " << options.output_dir << std::endl;
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
      LOG(INFO) << "Starting at frame " << options.start_index << std::endl;
      iterator_ptr->SetInitFrame(options.start_index);
    }
    while (iterator_ptr->HasNext() && (options.max_frames < 0 || frame_id < options.max_frames)) {
      auto time_start_frame = std::chrono::steady_clock::now();
      std::vector<Point3D> frame = iterator_ptr->Next();

      auto time_read_pointcloud = std::chrono::steady_clock::now();

      auto summary = ct_icp_odometry.RegisterFrame(frame);
      avg_number_of_attempts += summary.number_of_attempts;
      auto time_register_frame = std::chrono::steady_clock::now();

      std::chrono::duration<double> total_elapsed = time_register_frame - time_start_frame;
      std::chrono::duration<double> registration_elapsed = time_register_frame - time_read_pointcloud;

      registration_elapsed_ms += registration_elapsed.count() * 1000;
      all_seq_registration_elapsed_ms += registration_elapsed.count() * 1000;

      if (!summary.success) {
        LOG(ERROR) << "Error while running SLAM for sequence " << sequence_id << ", at frame index " << frame_id
                   << ". Error Message: " << summary.error_message << std::endl;
        if (options.suspend_on_failure) {
          exit(1);
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
          exit(1);
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

      LOG(INFO) << "[RESULTS] Sequence " << _sequence_name << std::endl;
      if (!valid_trajectory) {
        LOG(INFO) << "Invalid Trajectory, Failed after " << ground_truth_poses.size() << std::endl;
        LOG(INFO) << "Num Poses : " << seq_error.mean_rpe << std::endl;
      }
      LOG(INFO) << "Average Number of Attempts : " << avg_number_of_attempts << std::endl;
      LOG(INFO) << "Mean RPE : " << seq_error.mean_rpe << std::endl;
      LOG(INFO) << "Mean APE : " << seq_error.mean_ape << std::endl;
      LOG(INFO) << "Max APE : " << seq_error.max_ape << std::endl;
      LOG(INFO) << "Mean Local Error : " << seq_error.mean_local_err << std::endl;
      LOG(INFO) << "Max Local Error : " << seq_error.max_local_err << std::endl;
      LOG(INFO) << "Index Max Local Error : " << seq_error.index_max_local_err << std::endl;
      LOG(INFO) << "Average Duration : " << registration_elapsed_ms / frame_id << std::endl;
      LOG(INFO) << std::endl;

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
    LOG(INFO) << std::endl;
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
    LOG(INFO) << "KITTI metric translation/rotation : " << (all_seq_rpe_t / num_total_errors) * 100 << " "
              << (all_seq_rpe_r / num_total_errors) * 180.0 / M_PI << std::endl;
    LOG(INFO) << "Average RPE on seq : " << average_rpe_on_seq / nb_seq_with_gt;
  }

  LOG(INFO) << std::endl;
  LOG(INFO) << "Average registration time for all sequences (ms) : "
            << all_seq_registration_elapsed_ms / all_seq_num_frames << std::endl;

  // rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}