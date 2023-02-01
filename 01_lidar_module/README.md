# Lidar Obstacle Detection Project



### I. Preparation on Ubuntu

env: ubuntu18.04、pcl-1.8
```bash
$ sudo apt install libpcl-dev
$ git clone https://github.com/yaodix/sensor_fusion.git  ./01_lidar_module
```

The directory structure of the starter code looks like:
- src
    * render
        + `box.h`: struct definition for box objects
        + `render.h`, `render.cpp`: classes and methods for rendering objects
    * sensors
        + `lidar.h`: functions using ray casting for creating PCD
    * `environment.cpp`: main file for using PCL viewer and processing/visualizing PCD
    * `processPointClouds.h`, `processPointClouds.cpp`: functions for filtering, segmenting, clustering, boxing, loading and saving PCD

Try compiling the lidar simulator and run it. A window with a simulated highway environment should pop up.
```bash
$ cd 01_lidar_module
$ mkdir build && cd build
$ cmake ..
$ make
$ ./environment
```



### II. Exercises

#### Lidar and Point Clouds

1. Create a `Lidar` pointer object on the heap, taking two parameters: `std::vector<Car>` and `setGroundSlope` of 0. The `Lidar::scan()` method does a ray casting and returns a point cloud pointer object `pcl::PointCloud<pcl::PointXYZ>::Ptr`. Call `renderRays()` to plot rays on the viewer. 

<img src="media/first-lidar-object.png" width="800" height="400" />

2. Increase the lidar resolution by tweaking the constructor of `Lidar` class: `numLayers`, `horizontalAngleInc`. Set `minDistance` to 5 meter to remove points from the vehicle's rooftop. Set `sderr` to 0.2 to add some noises to the PCD. 

<img src="media/increase-lidar-resolution.png" width="800" height="400" />

3. Remove rendering for the highway scene and rays, but enable rendering for the point cloud using `renderPointCloud()`. 

<img src="media/render-point-cloud.png" width="800" height="400" />

#### Point Cloud Segmentation

1. In the `processPointCloud.cpp`, to implement the `ProcessPointClouds::SegmentPlane()` method, use a `pcl::SACSegmentation<PointT>` object to segment the planar component from the input point cloud. Next to implement the `SeparateClouds()` helper method, use a `pcl::ExtractIndices<PointT>` object to extract the points not belong to the plane as the obstacles. 

2. In the `environment.cpp`, create a `ProcessPointClouds<pcl::PointXYZ>` object, call `SegmentPlane()` to separate the plane and obstacle. Finally, render the plane and obstacle point clouds. 

<img src="media/segment-plane.png" width="800" height="400" />

#### Identify Different Obstacles using Euclidean Clustering with PCL

1. In the `processPointCloud.cpp`, to implement the `ProcessPointClouds::Clustering()` method, create a Kd-tree representation `pcl::search::KdTree<PointT>::Ptr` for the input point cloud, configure the parameters for the Euclidean clustering object `pcl::EuclideanClusterExtraction<PointT>` and extract the clusters in the point cloud. In the `environment.cpp`, call the clustering function on the segmented obstacle point cloud, render clustered obstacle in different colors. 

<img src="media/clustering-objects.png" width="800" height="400" />

#### Bounding Boxes

1. Once point cloud clusters are found, we can add bounding boxes around the clusters. The boxed spaces should be considered as an area/object that our car is not allowed to enter, otherwise it would result a collision. Call `ProcessPointClouds::BoundingBox()` method, which finds the max and min point values as the boundary values for the `Box` data structure. Then render the `Box` structure for each cluster. 
<img src="media/bounding-boxes.png" width="800" height="400" />

#### Load Real PCD

1. Create a new point processor for the real PCD from a `cityBlock`, the code is similar to the `simpleHighway` function. The point type is `pcl::PointXYZI` where the `I` indicates the intensity. The real PCD files are located at `src/sensors/data/pcd/data_1/` directory. 6d28e82))

<img src="media/load-real-pcd.png" width="800" height="400" />

#### Filter with PCL

1. To implement the `ProcessPointClouds::FilterCloud()` in the `processPointClouds.cpp`, `pcl::VoxelGrid<PointT>` class is applied for **Voxel Grid** filtering, and `pcl::CropBox<PointT>` class is applied for **ROI-based** filtering. The `Eigen::Vector4f` class has four parameters representing `x`, `y`, `z` coordinates and the last one should be 1.0. We are interested in a good amount of distance in front or at back of the car and surroundings of the car. Point cloud data outside of the ROI should be removed, including the rooftop points. 

2. In the `environment.cpp`, call `ProcessPointClouds::FilterCloud()` function in the `cityBlock()`. Input a leaf size of 0.2m, so that the voxel size is large enough to help speed up the processing but not so large that object definition is preserved.

<img src="media/downsampled-cloud.png" width="800" height="400" />

#### Obstacle Detection with Real PCD

1. Once having a filtered PCD, we can deploy the same segmentation and clustering techniques implemented previously in the `cityBlock()`. 

2. Tweak the `Eigen::Vector4f minPoint/maxPoint` for `ProcessPointClouds::FilterCloud()` and `int minSize/maxSize` for `ProcessPointClouds::Clustering()`. 

<img src="media/obstacle-detection-with-real-pcd.png" width="800" height="400" />

#### Stream PCD

1. Create a vector `stream` to store paths to the PCD files chronologically. Create a new `cityBlock()` function, which processes the input point cloud from the external. In the `main()` function of `environment.cpp`, inside the viewer update loop, read PCD file, process it and update the viewer. 

<img src="media/stream-pcd.gif" width="800" height="400" />



### III. Lidar Obstacle Detection Project

The previous exercises have created a processing pipeline for detecting the obstacles using PCL segmentation and clustering methods. In the final project, we have to implement the same obstacle detection pipeline but with the *3D RANSAC segmentation*, *KD-Tree*, and *Euclidean clustering algorithm* we created in the quizzes.

1. Create a `ProcessPointClouds::SegmentPlaneRansac()` function in the `processPointClouds.cpp`, which uses the 3D RANSAC segmentation for plane implemented in the quiz. And call this segmentation function in the `cityBlock()` of `environment.cpp`. 

2. Create `ProcessPointClouds::ClusteringEuclidean` and its helper function `ProcessPointClouds::clusterHelper` in the `processPointClouds.cpp`, which reuses the Euclidean clustering with KD-Tree with some modification from the quiz. And call this clustering function in the `cityBlock()` of `environment.cpp`. 

3. Before building the final project, make sure the macro `CUSTOM_METHOD` is defined in the `environment.cpp` so that the custom implementations in the Step 1 and 2 are used. Otherwise, defining macro `PCL_METHOD` will use the segmentation and clustering functions provided by PCL library.

<img src="media/obstacle-detection-fps-final.gif" width="800" height="400" />



### IV. Tracking and Challenge Problem



### V. References

`pcl::PointXYZ`: https://pointclouds.org/documentation/structpcl_1_1_point_x_y_z.html

Example of segmenting the Plane with PCL: https://pointclouds.org/documentation/tutorials/extract_indices.html

Example of Euclidean Cluster Extraction: https://pointclouds.org/documentation/tutorials/cluster_extraction.html

Example of Voxel Grid filtering: https://pointclouds.org/documentation/tutorials/voxel_grid.html

Example of Crop Box filtering: https://pointclouds.org/documentation/classpcl_1_1_crop_box.html