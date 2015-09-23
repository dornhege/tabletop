/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// Author(s): Christian Dornhege, Marius Muja, Matei Ciocarlie and Romain Thibaux

#ifndef _ICP_FITTER_H_
#define _ICP_FITTER_H_

#include <tabletop_object_detector/model_fitter.h>
#include <boost/function.hpp>
#include <ros/ros.h>
#include <opencv2/flann/flann.hpp>

namespace tabletop_object_detector {

//! Does an ICP fitting
class IcpFitter : public DistanceFieldFitter
{
 private:
  std::string sensor_frame_id_;

  int min_iterations_;
  int max_iterations_;
  double min_iteration_improvement_;

  double downsample_leaf_size_;

  // for ICP correspondences to consider
  double outlier_dist_;
  std::string outlier_kernel_;
  // for computing the match score
  double inlier_dist_;
  std::string inlier_kernel_;

  ros::Publisher pubMarker;

  //! Helper function for fitting
  cv::Point3f centerOfSupport(const std::vector<cv::Vec3f>& cloud) const;

  Eigen::Affine3d computeLocalTransform(const Eigen::Matrix3d & W, const Eigen::Vector3d cloudMu,
          const Eigen::Vector3d distance_voxel_grid_Mu) const;

  //! Inner loop when doing translation fitting
  double getFitScoreAndGradient(const std::vector<cv::Vec3f>& cloud,
                                const cv::Point3f& location, cv::Point3f& vector,
                                boost::function<double(double)> kernel) const;

  double getModelFitScore(const std::vector<cv::Vec3f>& cloud, const Eigen::Affine3d & pose,
                          boost::function<double(double)> kernel, cv::flann::Index& search,
                          const geometry_msgs::Pose & cloud_pose) const;

  visualization_msgs::Marker createMeshMarker(int id,
        const geometry_msgs::Pose & cloud_pose, const Eigen::Affine3d & icp_transform,
        const std::string & ns) const;

  visualization_msgs::Marker createClusterMarker(const EigenSTL::vector_Vector3d & cluster, int id,
        const geometry_msgs::Pose & cloud_pose,
        const std::string & ns) const;

  double applyTransformAndcomputeScore(EigenSTL::vector_Vector3d & cloud,
        const Eigen::Affine3d & transform,
        boost::function<double(double)> distance_score_kernel) const;

  Eigen::Matrix3d computeW(
        const EigenSTL::vector_Vector3d & cloud, const Eigen::Vector3d & cloudMu,
        const Eigen::Vector3d & distance_voxel_grid_Mu,
        boost::function<double(double)> distance_selection_kernel) const;

  void computeMus(const EigenSTL::vector_Vector3d & cloud,
          Eigen::Vector3d & cloudMu, Eigen::Vector3d & distance_voxel_grid_Mu,
         boost::function<double(double)> distance_selection_kernel) const;

  static inline double selectionKernel(double clip, double x)
  {
      if(x <= clip)
          return 1.0;
      return 0.0;
  }

  static inline double huberKernel(double clipping, double x)
  {
      if (x < clipping)
          return 1.0;
      else
          return (clipping / x);
  }

 protected:
  // do normal 3d ICP or restrict to 2d transforms
  bool use_3d_icp_;

 public:
  //! Stub, just calls super's constructor
  IcpFitter() : DistanceFieldFitter(), use_3d_icp_(true) {
    ros::NodeHandle nhPriv("~");
    if(!nhPriv.getParam("sensor_frame", sensor_frame_id_)) {
        ROS_WARN("%s: Could not get parameter for sensor_frame.", __PRETTY_FUNCTION__);
    }

    nhPriv.param("min_iterations", min_iterations_, 20);
    nhPriv.param("max_iterations", max_iterations_, 1000);
    nhPriv.param("min_iteration_improvement", min_iteration_improvement_, 0.001);

    nhPriv.param("downsample_leaf_size", downsample_leaf_size_, 0.003);

    nhPriv.param("outlier_dist", outlier_dist_, 0.02);
    nhPriv.param("outlier_kernel", outlier_kernel_, std::string("selection"));
    nhPriv.param("inlier_dist", inlier_dist_, 0.002);
    nhPriv.param("inlier_kernel", inlier_kernel_, std::string("huber"));

    ros::NodeHandle nh;
    pubMarker = nh.advertise<visualization_msgs::MarkerArray>("object_detection_marker", 10);
  }

  //! Empty stub
  ~IcpFitter() {}

  //! Main fitting function
  ModelFitInfo fitPointCloud(const std::vector<cv::Vec3f>& cloud, const geometry_msgs::Pose & cloud_pose,
          cv::flann::Index &search, double min_object_score) const;
};

class IcpFitter2d : public IcpFitter
{
    public:
        IcpFitter2d() : IcpFitter()
        {
            use_3d_icp_ = false;
        }
};

} //namespace

#endif

