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

// Author(s): Marius Muja, Matei Ciocarlie and Romain Thibaux

#ifndef _ITERATIVE_DISTANCE_FITTER_H_
#define _ITERATIVE_DISTANCE_FITTER_H_

#include <tabletop_object_detector/model_fitter.h>
#include <boost/function.hpp>
#include <ros/ros.h>
#include <opencv2/flann/flann.hpp>

namespace tabletop_object_detector {

// use M-kernel to weight inliers and suppress the influence of outliers from linear to just constant ->
// 1) more robust ICP (not too sensitive to outliers)
// 2) score = inliers, but as a floating point value -> no cut off threshold -> smooth -> can better distinguish between similar poses.
inline double huberKernel (double clipping, double x)
{
  if (x < clipping)
    return 1.0;
  else
    return (clipping / x);
}

//! Does an ICP-like fitting only in the X and Y translation DOFs
class IterativeTranslationFitter : public DistanceFieldFitter
{
 private:
  double clipping_;

  //! Helper function for fitting
  cv::Point3f centerOfSupport(const std::vector<cv::Vec3f>& cloud) const;

  //! Inner loop when doing translation fitting
  double getFitScoreAndGradient(const std::vector<cv::Vec3f>& cloud,
                                const cv::Point3f& location, cv::Point3f& vector,
                                boost::function<double(double)> kernel) const;

  double getModelFitScore(const std::vector<cv::Vec3f>& cloud, const cv::Point3f& location,
                          boost::function<double(double)> kernel, cv::flann::Index& search) const;

  visualization_msgs::Marker createClusterMarker(const EigenSTL::vector_Vector3d & cluster, int id,
        const geometry_msgs::Pose & cloud_pose, const Eigen::Affine3d & icp_transform) const;

  double applyTransformAndcomputeScore(EigenSTL::vector_Vector3d & cloud,
        const Eigen::Affine3d & transform,
        boost::function<double(double)> distance_score_kernel) const;

  Eigen::Matrix3d computeW(
        const EigenSTL::vector_Vector3d & cloud, const Eigen::Vector3d & cloudMu,
        const Eigen::Vector3d & distance_voxel_grid_Mu,
        boost::function<double(double)> distance_score_kernel) const;

  void computeMus(const EigenSTL::vector_Vector3d & cloud,
          Eigen::Vector3d & cloudMu, Eigen::Vector3d & distance_voxel_grid_Mu,
          boost::function<double(double)> distance_score_kernel) const;

 public:
  //! Stub, just calls super's constructor
  IterativeTranslationFitter() : DistanceFieldFitter() {
    ros::NodeHandle nhPriv("~");
    nhPriv.param("clipping", clipping_, 0.0075);
  }

  //! Empty stub
  ~IterativeTranslationFitter() {}

  //! Main fitting function
  ModelFitInfo fitPointCloud(const std::vector<cv::Vec3f>& cloud, cv::flann::Index &search,
                             double min_object_score, const geometry_msgs::Pose & cloud_pose) const;
};

} //namespace

#endif

