/* obstacle_detector.hpp

 * Copyright (C) 2021 SS47816

 * Implementation of 3D LiDAR Obstacle Detection & Tracking Algorithms

**/

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "box.hpp"

namespace lidar_obstacle_detector
{
template <typename PointT>
class ObstacleDetector
{
 public:
  ObstacleDetector();
  virtual ~ObstacleDetector();
  
  // ****************** Detection ***********************
  std::map<int, int> countObjectOccurrences(const std::vector<Box>& boxes) {
    std::map<int, int> objectCounts;

    for (const auto& box : boxes) {
        objectCounts[box.id]++;
    }
  return objectCounts;
  }
  typename pcl::PointCloud<PointT>::Ptr filterCloud(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const float filter_res, const Eigen::Vector4f& min_pt, const Eigen::Vector4f& max_pt);
  
  std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentPlane(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const int max_iterations, const float distance_thresh);

  std::vector<typename pcl::PointCloud<PointT>::Ptr> clustering(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const float cluster_tolerance, const int min_size, const int max_size);

  Box axisAlignedBoundingBox(const typename pcl::PointCloud<PointT>::ConstPtr& cluster, const int id);

  Box pcaBoundingBox(typename pcl::PointCloud<PointT>::Ptr& cluster, const int id);

  // ****************** Tracking ***********************
  void obstacleTracking(const std::vector<Box>& prev_boxes, std::vector<Box>& curr_boxes, const float displacement_thresh, const float iou_thresh);

 private:
  // ****************** Detection ***********************
  std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> separateClouds(const pcl::PointIndices::ConstPtr& inliers, const typename pcl::PointCloud<PointT>::ConstPtr& cloud);

  // ****************** Tracking ***********************
  bool compareBoxes(const Box& a, const Box& b, const float displacement_thresh, const float iou_thresh);

  // Link nearby bounding boxes between the previous and previous frame
  std::vector<std::vector<int>> associateBoxes(const std::vector<Box>& prev_boxes, const std::vector<Box>& curr_boxes, const float displacement_thresh, const float iou_thresh);

  // Connection Matrix
  std::vector<std::vector<int>> connectionMatrix(const std::vector<std::vector<int>>& connection_pairs, std::vector<int>& left, std::vector<int>& right);

  // Helper function for Hungarian Algorithm
  bool hungarianFind(const int i, const std::vector<std::vector<int>>& connection_matrix, std::vector<bool>& right_connected, std::vector<int>& right_pair);

  // Customized Hungarian Algorithm
  std::vector<int> hungarian(const std::vector<std::vector<int>>& connection_matrix);

  // Helper function for searching the box index in boxes given an id
  int searchBoxIndex(const std::vector<Box>& Boxes, const int id);
};

// constructor:
template <typename PointT>
ObstacleDetector<PointT>::ObstacleDetector() {}

// de-constructor:
template <typename PointT>
ObstacleDetector<PointT>::~ObstacleDetector() {}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ObstacleDetector<PointT>::filterCloud(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const float filter_res, const Eigen::Vector4f& min_pt, const Eigen::Vector4f& max_pt)
{
  // Time segmentation process
  // const auto start_time = std::chrono::steady_clock::now();

  // Create the filtering object: downsample the dataset using a leaf size
  pcl::VoxelGrid<PointT> vg;
  typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
  vg.setInputCloud(cloud);
  vg.setLeafSize(filter_res, filter_res, filter_res);
  vg.filter(*cloud_filtered);

  // Cropping the ROI
  typename pcl::PointCloud<PointT>::Ptr cloud_roi(new pcl::PointCloud<PointT>);
  pcl::CropBox<PointT> region(true);
  region.setMin(min_pt);
  region.setMax(max_pt);
  region.setInputCloud(cloud_filtered);
  region.filter(*cloud_roi);

  // Removing the car roof region
  std::vector<int> indices;
  pcl::CropBox<PointT> roof(true);
  roof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
  roof.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
  roof.setInputCloud(cloud_roi);
  roof.filter(indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for (auto& point : indices)
    inliers->indices.push_back(point);

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud_roi);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud_roi);

  // const auto end_time = std::chrono::steady_clock::now();
  // const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "filtering took " << elapsed_time.count() << " milliseconds" << std::endl;

  return cloud_roi;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ObstacleDetector<PointT>::separateClouds(const pcl::PointIndices::ConstPtr& inliers, const typename pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  typename pcl::PointCloud<PointT>::Ptr obstacle_cloud(new pcl::PointCloud<PointT>());
  typename pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>());

  // Pushback all the inliers into the ground_cloud
  for (int index : inliers->indices)
  {
    ground_cloud->points.push_back(cloud->points[index]);
  }

  // Extract the points that are not in the inliers to obstacle_cloud
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*obstacle_cloud);

  return std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>(obstacle_cloud, ground_cloud);
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ObstacleDetector<PointT>::segmentPlane(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const int max_iterations, const float distance_thresh)
{
  // Time segmentation process
  // const auto start_time = std::chrono::steady_clock::now();

  pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
  if (inliers->indices.empty())
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  return separateClouds(inliers, cloud);
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ObstacleDetector<PointT>::clustering(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const float cluster_tolerance, const int min_size, const int max_size)
{

  const int region_max_ = 2; 
  int regions_[100];
  regions_[0] = 4; regions_[1] = 14; regions_[2] = 14; regions_[3] = 14; regions_[4] = 14;
  float radius_min_ = 0;
  float radius_max_ = 10;
  int cluster_size_min_ = 20;
  int cluster_size_max_ = 10000;
  float k_merging_threshold_ = 0.1;
  float z_axis_min_ = 1.5;
  float z_axis_max_ = 1.5;
  // Time clustering process
  // const auto start_time = std::chrono::steady_clock::now();

  //std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
      /*** Downsampling + ground & ceiling removal ***/
  pcl::IndicesPtr pc_indices(new std::vector<int>);

  for(int i = 0; i < cloud->size(); ++i) {
    if(cloud->points[i].z >= z_axis_min_) {
        pc_indices->push_back(i);
    }
  }
     /*** Divide the point cloud into nested circular regions ***/
    boost::array<std::vector<int>, region_max_> indices_array;
    for(int i = 0; i < pc_indices->size(); i++) {
      
      float range = 0.0;
      
      for(int j = 0; j < region_max_; j++) {
        
        float d2 = cloud->points[(*pc_indices)[i]].x * cloud->points[(*pc_indices)[i]].x +
        cloud->points[(*pc_indices)[i]].y * cloud->points[(*pc_indices)[i]].y +
        cloud->points[(*pc_indices)[i]].z * cloud->points[(*pc_indices)[i]].z;
        
        if(d2 > radius_min_ * radius_min_ && d2 < radius_max_ * radius_max_ &&
          d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
            indices_array[j].push_back((*pc_indices)[i]);
            break;
        }
        range += regions_[j];
      }
    }
    
    /*** Euclidean clustering ***/
    float tolerance = 0.1;
    float adaptive_tolerance = cluster_tolerance;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    int last_clusters_begin = 0;
    int last_clusters_end = 0.1;

    for(int i = 0; i < region_max_; i++) {

      tolerance = tolerance + cluster_tolerance;
      if(indices_array[i].size() > cluster_size_min_) {
        boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud, indices_array_ptr);
        
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(tolerance);
        ec.setMinClusterSize(cluster_size_min_);
        ec.setMaxClusterSize(cluster_size_max_);   
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.setIndices(indices_array_ptr);
        ec.extract(cluster_indices);


        for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      
          for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            cluster->points.push_back(cloud->points[*pit]);
          }

    for(int j = last_clusters_begin; j < last_clusters_end; j++) {
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      int K = 1; //the number of neighbors to search for
      std::vector<int> k_indices(K);
      std::vector<float> k_sqr_distances(K);
      kdtree.setInputCloud(cluster);
      if(clusters[j]->points.size() >= 1) {
        if(kdtree.nearestKSearch(*clusters[j], clusters[j]->points.size()-1, K, k_indices, k_sqr_distances) > 0) {
          if(k_sqr_distances[0] < k_merging_threshold_) {
      *cluster += *clusters[j];
      clusters.erase(clusters.begin()+j);
      last_clusters_end--;
      //std::cerr << "k-merging: clusters " << j << " is merged" << std::endl; 
          }
        }
      }
    }

    /**************************************************/
          cluster->width = cluster->size();
          cluster->height = 1;
          cluster->is_dense = true;
    clusters.push_back(cluster);
        }
        /*** Merge z-axis clusters ***/
        for(int j = last_clusters_end; j < clusters.size(); j++) {
          Eigen::Vector4f j_min, j_max;
          pcl::getMinMax3D(*clusters[j], j_min, j_max);
          for(int k = j+1; k < clusters.size(); k++) {
            Eigen::Vector4f k_min, k_max;

            Eigen::Vector4f point_b ;
            Eigen::Vector4f point_a;
            pcl::compute3DCentroid(*clusters[k], point_a);
            pcl::compute3DCentroid(*clusters[j], point_b);;
            double distance = sqrt(pow(point_b[0] - point_a[0], 2) + pow(point_b[1] - point_a[1], 2) + pow(point_b[2] - point_a[2], 2));
            double dist_thrsh;
            if (ros::this_node::getName() == "/environment_segmentation") {
                dist_thrsh = 0.8;
            } else {
                dist_thrsh = 0.2;
            }

            if (distance < dist_thrsh && (clusters[j]->size() < 300 || clusters[k]->size() < 300)){
              *clusters[j] += *clusters[k];
              clusters.erase(clusters.begin()+k);
            }
            //if (clusters[k]->size() > 7500 || clusters[k]->size() < 75){clusters.erase(clusters.begin()+k);}
          }
        }
        //last_clusters_begin = last_clusters_end;
        last_clusters_end = 0;
      }
    }

  return clusters;
}

template <typename PointT>
Box ObstacleDetector<PointT>::axisAlignedBoundingBox(const typename pcl::PointCloud<PointT>::ConstPtr& cluster, const int id)
{
  // Find bounding box for one of the clusters
  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cluster, min_pt, max_pt);
  
  const Eigen::Vector3f position((max_pt.x + min_pt.x)/2, (max_pt.y + min_pt.y)/2, (max_pt.z + min_pt.z)/2);
  const Eigen::Vector3f dimension((max_pt.x - min_pt.x), (max_pt.y - min_pt.y), (max_pt.z - min_pt.z));

  return Box(id, position, dimension);
}

template <typename PointT>
Box ObstacleDetector<PointT>::pcaBoundingBox(typename pcl::PointCloud<PointT>::Ptr& cluster, const int id)
{
  // Compute the bounding box height (to be used later for recreating the box)
  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cluster, min_pt, max_pt);
  const float box_height = max_pt.z - min_pt.z;
  const float box_z = (max_pt.z + min_pt.z)/2;

  // Compute the cluster centroid 
  Eigen::Vector4f pca_centroid;
  pcl::compute3DCentroid(*cluster, pca_centroid);

  // Squash the cluster to x-y plane with z = centroid z 
  for (size_t i = 0; i < cluster->size(); ++i)
  {
    cluster->points[i].z = pca_centroid(2);
  }

  // Compute principal directions & Transform the original cloud to PCA coordinates
  pcl::PointCloud<pcl::PointXYZ>::Ptr pca_projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cluster);
  pca.project(*cluster, *pca_projected_cloud);
  
  const auto eigen_vectors = pca.getEigenVectors();

  // Get the minimum and maximum points of the transformed cloud.
  pcl::getMinMax3D(*pca_projected_cloud, min_pt, max_pt);
  const Eigen::Vector3f meanDiagonal = 0.5f * (max_pt.getVector3fMap() + min_pt.getVector3fMap());

  // Final transform
  const Eigen::Quaternionf quaternion(eigen_vectors); // Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
  const Eigen::Vector3f position = eigen_vectors * meanDiagonal + pca_centroid.head<3>();
  const Eigen::Vector3f dimension((max_pt.x - min_pt.x), (max_pt.y - min_pt.y), box_height);

  return Box(id, position, dimension, quaternion);
}

// ************************* Tracking ***************************
template <typename PointT>
void ObstacleDetector<PointT>::obstacleTracking(const std::vector<Box>& prev_boxes, std::vector<Box>& curr_boxes, const float displacement_thresh, const float iou_thresh)
{
  // Tracking (based on the change in size and displacement between frames)
  
  if (curr_boxes.empty() || prev_boxes.empty())
  {
    return;
  }
  else
  {
    // vectors containing the id of boxes in left and right sets
    std::vector<int> pre_ids;
    std::vector<int> cur_ids;
    std::vector<int> matches;

    // Associate Boxes that are similar in two frames
    auto connection_pairs = associateBoxes(prev_boxes, curr_boxes, displacement_thresh, iou_thresh);

    if (connection_pairs.empty()) return;

    // Construct the connection matrix for Hungarian Algorithm's use
    auto connection_matrix = connectionMatrix(connection_pairs, pre_ids, cur_ids);

    // Use Hungarian Algorithm to solve for max-matching
    matches = hungarian(connection_matrix);
    
    for (int j = 0; j < matches.size(); ++j){
      // find the index of the previous box that the current box corresponds to
      const auto pre_id = pre_ids[matches[j]];
      const auto pre_index = searchBoxIndex(prev_boxes, pre_id);
      
      // find the index of the current box that needs to be changed
      const auto cur_id = cur_ids[j]; // right and matches has the same size
      const auto cur_index = searchBoxIndex(curr_boxes, cur_id);
      
      if (pre_index > -1 && cur_index > -1)
      {
        // change the id of the current box to the same as the previous box
        curr_boxes[cur_index].id = prev_boxes[pre_index].id;
      }
    }
  }
}

template <typename PointT>
bool ObstacleDetector<PointT>::compareBoxes(const Box& a, const Box& b, const float displacement_thresh, const float iou_thresh)
{
  // Percetage Displacements ranging between [0.0, +oo]
  const float dis = sqrt((a.position[0] - b.position[0]) * (a.position[0] - b.position[0]) + (a.position[1] - b.position[1]) * (a.position[1] - b.position[1]) + (a.position[2] - b.position[2]) * (a.position[2] - b.position[2]));

  const float a_max_dim = std::max(a.dimension[0], std::max(a.dimension[1], a.dimension[2]));
  const float b_max_dim = std::max(b.dimension[0], std::max(b.dimension[1], b.dimension[2]));
  const float ctr_dis = dis / std::min(a_max_dim, b_max_dim);

  // Dimension similiarity values between [0.0, 1.0]
  const float x_dim = 2 * (a.dimension[0] - b.dimension[0]) / (a.dimension[0] + b.dimension[0]);
  const float y_dim = 2 * (a.dimension[1] - b.dimension[1]) / (a.dimension[1] + b.dimension[1]);
  const float z_dim = 2 * (a.dimension[2] - b.dimension[2]) / (a.dimension[2] + b.dimension[2]);

  if (ctr_dis <= displacement_thresh && x_dim <= iou_thresh && y_dim <= iou_thresh && z_dim <= iou_thresh)
  {
    return true;
  }
  else
  {
    return false;
  }
}

template <typename PointT>
std::vector<std::vector<int>> ObstacleDetector<PointT>::associateBoxes(const std::vector<Box>& prev_boxes, const std::vector<Box>& curr_boxes, const float displacement_thresh, const float iou_thresh)
{
  std::vector<std::vector<int>> connection_pairs;

  for (auto& prev_box : prev_boxes)
  {
    for (auto& curBox : curr_boxes)
    {
      // Add the indecies of a pair of similiar boxes to the matrix
      if (this->compareBoxes(curBox, prev_box, displacement_thresh, iou_thresh))
      {
        connection_pairs.push_back({prev_box.id, curBox.id});
      }
    }
  }

  return connection_pairs;
}

template <typename PointT>
std::vector<std::vector<int>> ObstacleDetector<PointT>::connectionMatrix(const std::vector<std::vector<int>>& connection_pairs, std::vector<int>& left, std::vector<int>& right)
{
  // Hash the box ids in the connection_pairs to two vectors(sets), left and right
  for (auto& pair : connection_pairs)
  {
    bool left_found = false;
    for (auto i : left)
    {
      if (i == pair[0])
        left_found = true;
    }
    if (!left_found)
      left.push_back(pair[0]);

    bool right_found = false;
    for (auto j : right)
    {
      if (j == pair[1])
        right_found = true;
    }
    if (!right_found)
      right.push_back(pair[1]);
  }

  std::vector<std::vector<int>> connection_matrix(left.size(), std::vector<int>(right.size(), 0));

  for (auto& pair : connection_pairs)
  {
    int left_index = -1;
    for (int i = 0; i < left.size(); ++i)
    {
      if (pair[0] == left[i])
        left_index = i;
    }

    int right_index = -1;
    for (int i = 0; i < right.size(); ++i)
    {
      if (pair[1] == right[i])
        right_index = i;
    }

    if (left_index != -1 && right_index != -1)
      connection_matrix[left_index][right_index] = 1;
  }

  return connection_matrix;
}

template <typename PointT>
bool ObstacleDetector<PointT>::hungarianFind(const int i, const std::vector<std::vector<int>>& connection_matrix, std::vector<bool>& right_connected, std::vector<int>& right_pair)
{
  for (int j = 0; j < connection_matrix[0].size(); ++j)
  {
    if (connection_matrix[i][j] == 1 && right_connected[j] == false)
    {
      right_connected[j] = true;

      if (right_pair[j] == -1 || hungarianFind(right_pair[j], connection_matrix, right_connected, right_pair))
      {
        right_pair[j] = i;
        return true;
      }
    }
  }
}

template <typename PointT>
std::vector<int> ObstacleDetector<PointT>::hungarian(const std::vector<std::vector<int>>& connection_matrix)
{
  std::vector<bool> right_connected(connection_matrix[0].size(), false);
  std::vector<int> right_pair(connection_matrix[0].size(), -1);

  int count = 0;
  for (int i = 0; i < connection_matrix.size(); ++i)
  {
    if (hungarianFind(i, connection_matrix, right_connected, right_pair))
      count++;
  }

  std::cout << "For: " << right_pair.size() << " current frame bounding boxes, found: " << count << " matches in previous frame! " << std::endl;

  return right_pair;
}

template <typename PointT>
int ObstacleDetector<PointT>::searchBoxIndex(const std::vector<Box>& boxes, const int id)
{
  for (int i = 0; i < boxes.size(); i++)
  {
    if (boxes[i].id == id)
    return i;
  }

  return -1;
}

}