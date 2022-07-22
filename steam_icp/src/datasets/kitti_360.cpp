#include "steam_icp/datasets/kitti_360.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "steam_icp/utils/ply_file.h"

namespace steam_icp {

namespace {

const std::vector<std::string> KITTI_360_SEQUENCE_NAMES = {"00", "02", "03", "04", "05", "06", "07", "09", "10"};

const int LENGTH_SEQUENCE_KITTI_360[] = {11500, 19230, 1029, 11399, 6722, 9697, 3160, 13954, 3742};

// Calibration
const double R_Tr_data_KITTI_360[] = {9.999290633685804508e-01, 5.805355888196038310e-03,  1.040029024212630118e-02,
                                      5.774300279226996999e-03, -9.999787876452227442e-01, 3.013573682642321436e-03,
                                      1.041756443854582707e-02, -2.953305511449066945e-03, -9.999413744330052367e-01};

const Eigen::Matrix3d R_Tr_KITTI_360(R_Tr_data_KITTI_360);
Eigen::Vector3d T_Tr_KITTI_360 =
    Eigen::Vector3d(-7.640302229235816922e-01, 2.966030253893782165e-01, -8.433819635885287935e-01);

inline std::string frame_file_name(int frame_id) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << frame_id;
  return "frame_" + ss.str() + ".ply";
}

std::vector<Point3D> readPointCloud(const std::string &path, const double &min_dist, const double &max_dist) {
  std::vector<Point3D> frame;
  // read ply frame file
  PlyFile plyFileIn(path, fileOpenMode_IN);
  char *dataIn = nullptr;
  int sizeOfPointsIn = 0;
  int numPointsIn = 0;
  plyFileIn.readFile(dataIn, sizeOfPointsIn, numPointsIn);

  // Specific Parameters for KITTI_raw
  const double KITTI_MIN_Z = -5.0;  // Bad returns under the ground
  const double KITTI_GLOBAL_VERTICAL_ANGLE_OFFSET =
      0.205;  // Issue in the intrinsic calibration of the KITTI Velodyne HDL64

  double frame_last_timestamp = 0.0;
  double frame_first_timestamp = 1000000000.0;
  frame.reserve(numPointsIn);
  for (int i(0); i < numPointsIn; i++) {
    unsigned long long int offset = (unsigned long long int)i * (unsigned long long int)sizeOfPointsIn;
    Point3D new_point;
    new_point.raw_pt[0] = *((float *)(dataIn + offset));
    offset += sizeof(float);
    new_point.raw_pt[1] = *((float *)(dataIn + offset));
    offset += sizeof(float);
    new_point.raw_pt[2] = *((float *)(dataIn + offset));
    offset += sizeof(float);
    new_point.pt = new_point.raw_pt;
    new_point.alpha_timestamp = *((float *)(dataIn + offset));
    offset += sizeof(float);

    if (new_point.alpha_timestamp < frame_first_timestamp) {
      frame_first_timestamp = new_point.alpha_timestamp;
    }

    if (new_point.alpha_timestamp > frame_last_timestamp) {
      frame_last_timestamp = new_point.alpha_timestamp;
    }

    double r = new_point.raw_pt.norm();
    if ((r > min_dist) && (r < max_dist) && (new_point.raw_pt[2] > KITTI_MIN_Z)) {
      frame.push_back(new_point);
    }
  }
  frame.shrink_to_fit();

  for (int i(0); i < (int)frame.size(); i++) {
    frame[i].alpha_timestamp = std::min(1.0, std::max(0.0, 1 - (frame_last_timestamp - frame[i].alpha_timestamp) /
                                                                   (frame_last_timestamp - frame_first_timestamp)));
  }
  delete[] dataIn;

  // Intrinsic calibration of the vertical angle of laser fibers (take the same correction for all lasers)
  for (int i = 0; i < (int)frame.size(); i++) {
    Eigen::Vector3d rotationVector = frame[i].pt.cross(Eigen::Vector3d(0., 0., 1.));
    rotationVector.normalize();
    Eigen::Matrix3d rotationScan;
    rotationScan = Eigen::AngleAxisd(KITTI_GLOBAL_VERTICAL_ANGLE_OFFSET * M_PI / 180.0, rotationVector);
    frame[i].raw_pt = rotationScan * frame[i].raw_pt;
    frame[i].pt = rotationScan * frame[i].pt;
  }
  return frame;
}

/* -------------------------------------------------------------------------------------------------------------- */
ArrayPoses loadPoses(const std::string &file_path) {
  ArrayPoses poses;
  std::ifstream pFile(file_path);
  if (pFile.is_open()) {
    while (!pFile.eof()) {
      std::string line;
      std::getline(pFile, line);
      if (line.empty()) continue;
      std::stringstream ss(line);
      Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
      ss >> P(0, 0) >> P(0, 1) >> P(0, 2) >> P(0, 3) >> P(1, 0) >> P(1, 1) >> P(1, 2) >> P(1, 3) >> P(2, 0) >>
          P(2, 1) >> P(2, 2) >> P(2, 3);
      poses.push_back(P);
    }
    pFile.close();
  } else {
    throw std::runtime_error{"unable to open file: " + file_path};
  }
  return poses;
}

/* -------------------------------------------------------------------------------------------------------------- */
ArrayPoses transformTrajectory(const Trajectory &trajectory) {
  // For KITTI_raw the evaluation counts the middle of the frame as the pose which is compared to the ground truth
  ArrayPoses poses;
  // denoting the rigid transformation from the first camera (image_00) to the Velodyne.
  Eigen::Matrix3d R_Tr = R_Tr_KITTI_360.transpose();
  // denoting the rigid transformation from the first camera (image_00) to the Velodyne.
  Eigen::Vector3d T_Tr = T_Tr_KITTI_360;
  Eigen::Matrix4d Tr = Eigen::Matrix4d::Identity();
  Tr.block<3, 3>(0, 0) = R_Tr;
  Tr.block<3, 1>(0, 3) = T_Tr;

  poses.reserve(trajectory.size());
  for (auto &frame : trajectory) {
    Eigen::Matrix3d center_R;
    Eigen::Vector3d center_t;
    Eigen::Quaterniond q_begin = Eigen::Quaterniond(frame.begin_R);
    Eigen::Quaterniond q_end = Eigen::Quaterniond(frame.end_R);
    Eigen::Vector3d t_begin = frame.begin_t;
    Eigen::Vector3d t_end = frame.end_t;
    Eigen::Quaterniond q = q_begin.slerp(0.5, q_end);
    q.normalize();
    center_R = q.toRotationMatrix();
    center_t = 0.5 * t_begin + 0.5 * t_end;

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = center_R;
    pose.block<3, 1>(0, 3) = center_t;

    // Transform the data into the left camera reference frame (left camera) and evaluate SLAM
    pose = Tr.inverse() * pose * Tr;

    poses.push_back(pose);
  }
  return poses;
}

/* -------------------------------------------------------------------------------------------------------------- */
inline double translationError(const Eigen::Matrix4d &pose_error) { return pose_error.block<3, 1>(0, 3).norm(); }
inline double translationError2D(const Eigen::Matrix4d &pose_error) { return pose_error.block<2, 1>(0, 3).norm(); }
inline double rotationError(Eigen::Matrix4d &pose_error) {
  double a = pose_error(0, 0);
  double b = pose_error(1, 1);
  double c = pose_error(2, 2);
  double d = 0.5 * (a + b + c - 1.0);
  return acos(std::max(std::min(d, 1.0), -1.0));
}
inline std::vector<double> trajectoryDistances(const ArrayPoses &poses) {
  std::vector<double> dist(1, 0.0);
  for (size_t i = 1; i < poses.size(); i++) dist.push_back(dist[i - 1] + translationError(poses[i - 1] - poses[i]));
  return dist;
}
inline int lastFrameFromSegmentLength(const std::vector<double> &dist, int first_frame, double len) {
  for (int i = first_frame; i < (int)dist.size(); i++)
    if (dist[i] > dist[first_frame] + len) return i;
  return -1;
}

void computeMeanRPE(const ArrayPoses &poses_gt, const ArrayPoses &poses_result, Sequence::SeqError &seq_err) {
  // static parameter
  double lengths[] = {100, 200, 300, 400, 500, 600, 700, 800};
  size_t num_lengths = sizeof(lengths) / sizeof(double);

  // parameters
  int step_size = 10;  // every 10 frame (= every second for LiDAR at 10Hz)

  // pre-compute distances (from ground truth as reference)
  std::vector<double> dist = trajectoryDistances(poses_gt);

  int num_total = 0;
  double mean_rpe = 0;
  double mean_rpe_2d = 0;
  // for all start positions do
  for (int first_frame = 0; first_frame < (int)poses_gt.size(); first_frame += step_size) {
    // for all segment lengths do
    for (size_t i = 0; i < num_lengths; i++) {
      // current length
      double len = lengths[i];

      // compute last frame
      int last_frame = lastFrameFromSegmentLength(dist, first_frame, len);

      // next frame if sequence not long enough
      if (last_frame == -1) continue;

      // compute translational errors
      Eigen::Matrix4d pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
      Eigen::Matrix4d pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
      Eigen::Matrix4d pose_error = pose_delta_result.inverse() * pose_delta_gt;
      double t_err = translationError(pose_error);
      double t_err_2d = translationError2D(pose_error);
      double r_err = rotationError(pose_error);
      seq_err.tab_errors.emplace_back(t_err / len, r_err / len);

      mean_rpe += t_err / len;
      mean_rpe_2d += t_err_2d / len;
      num_total++;
    }
  }

  seq_err.mean_rpe = ((mean_rpe / static_cast<double>(num_total)) * 100.0);
  seq_err.mean_rpe_2d = ((mean_rpe_2d / static_cast<double>(num_total)) * 100.0);
}

/* -------------------------------------------------------------------------------------------------------------- */
Sequence::SeqError eval(const ArrayPoses &poses_gt, const ArrayPoses &poses_estimated) {
  Sequence::SeqError seq_err;

  // Compute Mean and Max APE (Mean and Max Absolute Pose Error)
  seq_err.mean_ape = 0.0;
  seq_err.max_ape = 0.0;
  for (size_t i = 0; i < poses_gt.size(); i++) {
    double t_ape_err = translationError(poses_estimated[i].inverse() * poses_gt[i]);
    seq_err.mean_ape += t_ape_err;
    if (seq_err.max_ape < t_ape_err) {
      seq_err.max_ape = t_ape_err;
    }
  }
  seq_err.mean_ape /= static_cast<double>(poses_gt.size());

  // Compute Mean and Max Local Error
  seq_err.mean_local_err = 0.0;
  seq_err.max_local_err = 0.0;
  seq_err.index_max_local_err = 0;
  for (int i = 1; i < (int)poses_gt.size(); i++) {
    double t_local_err = fabs((poses_gt[i].block<3, 1>(0, 3) - poses_gt[i - 1].block<3, 1>(0, 3)).norm() -
                              (poses_estimated[i].block<3, 1>(0, 3) - poses_estimated[i - 1].block<3, 1>(0, 3)).norm());
    seq_err.mean_local_err += t_local_err;
    if (seq_err.max_local_err < t_local_err) {
      seq_err.max_local_err = t_local_err;
      seq_err.index_max_local_err = i;
    }
  }
  seq_err.mean_local_err /= static_cast<double>(poses_gt.size() - 1);

  // Compute sequence mean RPE errors
  computeMeanRPE(poses_gt, poses_estimated, seq_err);
  return seq_err;
}

}  // namespace

Kitti360Sequence::Kitti360Sequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/frames/";

  const auto itr = std::find(KITTI_360_SEQUENCE_NAMES.begin(), KITTI_360_SEQUENCE_NAMES.end(), options_.sequence);
  if (itr == KITTI_360_SEQUENCE_NAMES.end()) throw std::runtime_error{"unknow kitti sequence."};
  sequence_id_ = (int)std::distance(KITTI_360_SEQUENCE_NAMES.begin(), itr);

  last_frame_ = LENGTH_SEQUENCE_KITTI_360[sequence_id_] + 1;
  last_frame_ = std::min(last_frame_, options_.last_frame);
  curr_frame_ = std::max((int)0, options_.init_frame);
  init_frame_ = std::max((int)0, options_.init_frame);
  has_ground_truth_ = ((init_frame_ == 0) && last_frame_ == (LENGTH_SEQUENCE_KITTI_360[sequence_id_] + 1));
}

std::vector<Point3D> Kitti360Sequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = dir_path_ + frame_file_name(curr_frame);
  auto pc = readPointCloud(filename, options_.min_dist_lidar_center, options_.max_dist_lidar_center);
  for (auto &point : pc) point.timestamp = (static_cast<double>(curr_frame) + point.alpha_timestamp) / 10.0;
  return pc;
}

void Kitti360Sequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  const auto poses = transformTrajectory(trajectory);

  //
  const auto filename = path + "/" + options_.sequence + "_poses.txt";
  std::ofstream posefile(filename);
  if (!posefile.is_open()) throw std::runtime_error{"failed to open file: " + filename};
  posefile << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  for (auto &pose : poses) {
    R = pose.block<3, 3>(0, 0);
    t = pose.block<3, 1>(0, 3);
    posefile << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << t(0) << " " << R(1, 0) << " " << R(1, 1) << " "
             << R(1, 2) << " " << t(1) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t(2)
             << std::endl;
  }
}

auto Kitti360Sequence::evaluate(const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/" + options_.sequence + ".txt";
  const auto gt_poses = loadPoses(ground_truth_file);

  //
  const auto poses = transformTrajectory(trajectory);

  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  return eval(gt_poses, poses);
}

}  // namespace steam_icp