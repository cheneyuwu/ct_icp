#include "steam_icp/datasets/kitti_raw.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "steam_icp/utils/ply_file.h"

namespace steam_icp {

namespace {

const std::vector<std::string> KITTI_SEQUENCE_NAMES = {"00", "01", "02", "03", "04", "05", "06", "07",
                                                       "08", "09", "10", "11", "12", "13", "14", "15",
                                                       "16", "17", "18", "19", "20", "21"};

const int LENGTH_SEQUENCE_KITTI[] = {4540, 1100, 4660, 800, 270,  2760, 1100, 1100, 4070, 1590, 1200,
                                     920,  1060, 3280, 630, 1900, 1730, 490,  1800, 4980, 830,  2720};

// Calibration Sequence 00, 01, 02, 13, 14, 15, 16, 17, 18, 19, 20, 21
const double R_Tr_data_A_KITTI[] = {4.276802385584e-04,  -9.999672484946e-01, -8.084491683471e-03,
                                    -7.210626507497e-03, 8.081198471645e-03,  -9.999413164504e-01,
                                    9.999738645903e-01,  4.859485810390e-04,  -7.206933692422e-03};
Eigen::Matrix3d R_Tr_A_KITTI(R_Tr_data_A_KITTI);
Eigen::Vector3d T_Tr_A_KITTI = Eigen::Vector3d(-1.198459927713e-02, -5.403984729748e-02, -2.921968648686e-01);

// Calibration Sequence 03
const double R_Tr_data_B_KITTI[] = {2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02,
                                    1.044940741659e-02, 1.056535364138e-02,  -9.998895741176e-01,
                                    9.999453885620e-01, 1.243653783865e-04,  1.045130299567e-02};
const Eigen::Matrix3d R_Tr_B_KITTI(R_Tr_data_B_KITTI);
const Eigen::Vector3d T_Tr_B_KITTI = Eigen::Vector3d(-2.796816941295e-03, -7.510879138296e-02, -2.721327964059e-01);

// Calibration Sequence 04, 05, 06, 07, 08, 09, 10, 11, 12
const double R_Tr_data_C_KITTI[] = {-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03,
                                    -6.481465826011e-03, 8.051860151134e-03,  -9.999466081774e-01,
                                    9.999773098287e-01,  -1.805528627661e-03, -6.496203536139e-03};
const Eigen::Matrix3d R_Tr_C_KITTI(R_Tr_data_C_KITTI);
const Eigen::Vector3d T_Tr_C_KITTI = Eigen::Vector3d(-4.784029760483e-03, -7.337429464231e-02, -3.339968064433e-01);

const Eigen::Matrix3d R_Tr_array_KITTI[] = {
    R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_B_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI,
    R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_C_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI,
    R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI, R_Tr_A_KITTI};
const Eigen::Vector3d T_Tr_array_KITTI[] = {
    T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_B_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI,
    T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_C_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI,
    T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI, T_Tr_A_KITTI};

inline std::string frame_file_name(int frame_id) {
  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << frame_id;
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
ArrayPoses transformTrajectory(const Trajectory &trajectory, int id) {
  // For KITTI_raw the evaluation counts the middle of the frame as the pose which is compared to the ground truth
  ArrayPoses poses;
  Eigen::Matrix3d R_Tr = R_Tr_array_KITTI[id].transpose();
  Eigen::Vector3d T_Tr = T_Tr_array_KITTI[id];

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

    // Transform the data into the left camera reference frame (left camera) and evaluate SLAM
    center_R = R_Tr * center_R * R_Tr.transpose();
    center_t = -center_R * T_Tr + T_Tr + R_Tr * center_t;

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = center_R;
    pose.block<3, 1>(0, 3) = center_t;
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

KittiRawSequence::KittiRawSequence(const Options &options) : Sequence(options) {
  dir_path_ = options_.root_path + "/" + options_.sequence + "/frames/";

  const auto itr = std::find(KITTI_SEQUENCE_NAMES.begin(), KITTI_SEQUENCE_NAMES.end(), options_.sequence);
  if (itr == KITTI_SEQUENCE_NAMES.end()) throw std::runtime_error{"unknow kitti sequence."};
  sequence_id_ = (int)std::distance(KITTI_SEQUENCE_NAMES.begin(), itr);

  num_frames_ = LENGTH_SEQUENCE_KITTI[sequence_id_] + 1;
}

std::vector<Point3D> KittiRawSequence::next() {
  if (!hasNext()) throw std::runtime_error("No more frames in sequence");
  int curr_frame = curr_frame_++;
  auto filename = dir_path_ + frame_file_name(curr_frame);
  auto pc = readPointCloud(filename, options_.min_dist_lidar_center, options_.max_dist_lidar_center);
  for (auto &point : pc) point.timestamp = (static_cast<double>(curr_frame) + point.alpha_timestamp) / 10.0;
  return pc;
}

void KittiRawSequence::save(const std::string &path, const Trajectory &trajectory) const {
  //
  const auto poses = transformTrajectory(trajectory, sequence_id_);

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

auto KittiRawSequence::evaluate(const Trajectory &trajectory) const -> SeqError {
  //
  std::string ground_truth_file = options_.root_path + "/" + options_.sequence + "/" + options_.sequence + ".txt";
  const auto gt_poses = loadPoses(ground_truth_file);

  //
  const auto poses = transformTrajectory(trajectory, sequence_id_);

  //
  if (gt_poses.size() == 0 || gt_poses.size() != poses.size())
    throw std::runtime_error{"estimated and ground truth poses are not the same size."};

  return eval(gt_poses, poses);
}

}  // namespace steam_icp