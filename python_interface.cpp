#include <vector>
#include <random>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "codelibrary/geometry/io/xyz_io.h"
#include "codelibrary/geometry/point_cloud/pca_estimate_normals.h"
#include "codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "codelibrary/geometry/util/distance_3d.h"
#include "codelibrary/util/tree/kd_tree.h"
#include "include/vccs_supervoxel.h"
#include "include/vccs_knn_supervoxel.h"
#include "include/rgb2lab.h"


namespace py = pybind11;

struct ColoredPointWithNormal : cl::RPoint3D {
    ColoredPointWithNormal() {}

    cl::RVector3D normal;
    RGB rgb;
    const LAB get_lab() {
      return RGB2LAB(rgb);
    }
};

class VCCSMetric {
public:
    explicit VCCSMetric(double resolution, double spatial_importance = 0.4, double normal_importance = 1)
        : resolution_(resolution) , spatial_importance_(spatial_importance), normal_importance_(normal_importance) 
        
        {}

    double operator() (const ColoredPointWithNormal& p1,
                       const ColoredPointWithNormal& p2) const {
        // const size_t m2 = 16;
        // LAB l1 = RGB2LAB(p1.rgb);
        // LAB l2 = RGB2LAB(p2.rgb);
        // 0.4 * lab_dist(l1, l2) / m2;
        // py::print(p1.rgb.r, p1.rgb.b, p1.rgb.g, l1.l, l1.a, l1.b);
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) *  normal_importance_/ resolution_ * spatial_importance_;
    }

private:
    double resolution_;
    double spatial_importance_;
    double normal_importance_;
};

cl::Array<cl::RGB32Color> random_colors(cl::Array<cl::RPoint3D>& points, cl::Array<int> labels, int n_supervoxels) {
  // make some colors
  cl::Array<cl::RGB32Color> colors(points.size());
  std::mt19937 random;
  cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
  for (int i = 0; i < n_supervoxels; ++i) {
      supervoxel_colors[i] = cl::RGB32Color(random());
  }
  for (int i = 0; i < points.size(); ++i) {
      colors[i] = supervoxel_colors[labels[i]];
  }
  return colors;
}

py::array py_segment(py::array_t<double, py::array::c_style | py::array::forcecast>& array,
                      const float resolution,const float spatial_importance,const float normal_importance)
{
  // check input dimensions
  if ( array.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");
  if ( array.shape()[1] != 9 )
    throw std::runtime_error("Input should have size [N,9] i.e. [N, xyz rgb nxnynz]");

    unsigned int N = array.shape()[0];
    unsigned int W = array.shape()[1];
  std::vector<double> pos(array.size());
  // copy py::array -> std::vector
  std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

  // initialze first array for points for KDTree
  //py::print("copying ", N," points to cl::RPoint3d");
  cl::Array<cl::RPoint3D> points(N);
  for (int i = 0; i < N; i++) {
    // copy XYZ data into RPoints
    points[i].x = pos.data()[i * W + 0];
    points[i].y = pos.data()[i * W + 1];
    points[i].z = pos.data()[i * W + 2];
  }

  cl::KDTree<cl::RPoint3D> kdtree;
  kdtree.SwapPoints(&points);
  const int k_neighbors = 15;
  assert(k_neighbors < N);


  //py::print("Running KDTree.");
  //cl::Array<cl::RVector3D> normals(N);
  cl::Array<cl::Array<int> > neighbors(N);
  cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
  for (int i = 0; i < N; ++i) {
      kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                                   &neighbors[i]);
      for (int k = 0; k < k_neighbors; ++k) {
          neighbor_points[k] = kdtree.points()[neighbors[i][k]];
      }
      // we just use normals from scannet
      /*
      cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
                                                   neighbor_points.end(),
                                                   &normals[i]);
      */
  }
  kdtree.SwapPoints(&points);

  cl::Array<ColoredPointWithNormal> oriented_points(N);
  for (int i = 0; i < N; i++) {
    // copy XYZ data into RPoints
    oriented_points[i].x = pos.data()[i * W + 0];
    oriented_points[i].y = pos.data()[i * W + 1];
    oriented_points[i].z = pos.data()[i * W + 2];
    RGB rgb{};
    rgb.r = pos.data()[i * W + 3];
    rgb.g = pos.data()[i * W + 4];
    rgb.b = pos.data()[i * W + 5];
    oriented_points[i].rgb = rgb;
    cl::RVector3D normal{};
    normal.x = pos.data()[i * W + 6];
    normal.y = pos.data()[i * W + 7];
    normal.z = pos.data()[i * W + 8];
    //normal.x = normals[i].x;
    //normal.y = normals[i].y;
    //normal.z = normals[i].z;
    oriented_points[i].normal = normal;
  }

  //py::print("Calling supervoxel segmentation");
  VCCSMetric metric(resolution);
  cl::Array<int> labels, supervoxels;
  // LOG(INFO) << "Start supervoxel segmentation...";
  cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points,
                                                    neighbors,
                                                    resolution,
                                                    metric,
                                                    &supervoxels,
                                                    &labels);

  // call pure C++ function
  //py::gil_scoped_release release;
  //py::gil_scoped_acquire acquire;

  // TODO: replace this with actual pointcloud colors
  cl::Array<cl::RGB32Color> colors = random_colors(points, labels, supervoxels.size());

  // todo, we can allocate this explicitely
  std::vector<double> result;

 // py::print("Copying to output");
  for (int i = 0; i < oriented_points.size(); i++) {
    result.push_back(oriented_points[i].x);
    result.push_back(oriented_points[i].y);
    result.push_back(oriented_points[i].z);
    result.push_back(pos[i * W + 3]);
    result.push_back(pos[i * W + 4]);
    result.push_back(pos[i * W + 5]);
    result.push_back(static_cast<double>(colors[i].red()));
    result.push_back(static_cast<double>(colors[i].green()));
    result.push_back(static_cast<double>(colors[i].blue()));
    // add the label from the segmentation as output..
    result.push_back(static_cast<double>(labels[i]));
  }

  // returns same shape but with one additional point i.e. supervoxel cluster
  std::vector<unsigned int> shape   = { N, W + 1};
  std::vector<unsigned int> strides = { sizeof(double)*(W + 1) , sizeof(double) };
  //py::print("Returning values..");
  //py::print(result.size(), shape);

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    2,                                       /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
}

py::array py_segment_knn(py::array_t<double, py::array::c_style | py::array::forcecast>& array,
  const float resolution,const float spatial_importance,const float normal_importance)
{
// check input dimensions
if ( array.ndim()     != 2 )
throw std::runtime_error("Input should be 2-D NumPy array");
if ( array.shape()[1] != 9 )
throw std::runtime_error("Input should have size [N,9] i.e. [N, xyz rgb nxnynz]");

unsigned int N = array.shape()[0];
unsigned int W = array.shape()[1];
std::vector<double> pos(array.size());
// copy py::array -> std::vector
std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

// initialze first array for points for KDTree
//py::print("copying ", N," points to cl::RPoint3d");
cl::Array<cl::RPoint3D> points(N);
for (int i = 0; i < N; i++) {
// copy XYZ data into RPoints
points[i].x = pos.data()[i * W + 0];
points[i].y = pos.data()[i * W + 1];
points[i].z = pos.data()[i * W + 2];
}

cl::KDTree<cl::RPoint3D> kdtree;
kdtree.SwapPoints(&points);
const int k_neighbors = 15;
assert(k_neighbors < N);


//py::print("Running KDTree.");
// cl::Array<cl::RVector3D> normals(N);
cl::Array<cl::Array<int> > neighbors(N);
cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
for (int i = 0; i < N; ++i) {
  kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                  &neighbors[i]);
    for (int k = 0; k < k_neighbors; ++k) {
        neighbor_points[k] = kdtree.points()[neighbors[i][k]];
      }

// we just use normals from scannet

//cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
//                                neighbor_points.end(),
//                                &normals[i]);

}
//kdtree.SwapPoints(&points);
VCCSKNNSupervoxel vccs_knn(kdtree, resolution,spatial_importance,normal_importance);
//py::print("Calling supervoxel segmentation");
cl::Array<int> labels;
cl::Array<VCCSKNNSupervoxel::Supervoxel> supervoxels;
vccs_knn.Segment(&labels, &supervoxels);
//kdtree.SwapPoints(&points);
// call pure C++ function
//py::gil_scoped_release release;
//py::gil_scoped_acquire acquire;

// TODO: replace this with actual pointcloud colors
cl::Array<cl::RGB32Color> colors = random_colors(points, labels, supervoxels.size());

// todo, we can allocate this explicitely
std::vector<double> result;

// py::print("Copying to output");
for (int i = 0; i < points.size(); i++) {
result.push_back(points[i].x);
result.push_back(points[i].y);
result.push_back(points[i].z);
result.push_back(pos[i * W + 3]);
result.push_back(pos[i * W + 4]);
result.push_back(pos[i * W + 5]);
result.push_back(static_cast<double>(colors[i].red()));
result.push_back(static_cast<double>(colors[i].green()));
result.push_back(static_cast<double>(colors[i].blue()));
// add the label from the segmentation as output..
result.push_back(static_cast<double>(labels[i]));
}

// returns same shape but with one additional point i.e. supervoxel cluster
std::vector<unsigned int> shape   = { N, W + 1};
std::vector<unsigned int> strides = { sizeof(double)*(W + 1) , sizeof(double) };
//py::print("Returning values..");
//py::print(result.size(), shape);

// return 2-D NumPy array
return py::array(py::buffer_info(
result.data(),                           /* data as contiguous array  */
sizeof(double),                          /* size of one scalar        */
py::format_descriptor<double>::format(), /* data type                 */
2,                                       /* number of dimensions      */
shape,                                   /* shape of the matrix       */
strides                                  /* strides for each axis     */
));
}

py::array py_segment_vccs(py::array_t<double, py::array::c_style | py::array::forcecast>& array,
  const float resolution,const float voxel_resolution,const float spatial_importance,const float normal_importance)
{
// check input dimensions
if ( array.ndim()     != 2 )
throw std::runtime_error("Input should be 2-D NumPy array");
if ( array.shape()[1] != 9 )
throw std::runtime_error("Input should have size [N,9] i.e. [N, xyz rgb nxnynz]");

unsigned int N = array.shape()[0];
unsigned int W = array.shape()[1];
std::vector<double> pos(array.size());
// copy py::array -> std::vector
std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

// initialze first array for points for KDTree
//py::print("copying ", N," points to cl::RPoint3d");
cl::Array<cl::RPoint3D> points(N);
for (int i = 0; i < N; i++) {
// copy XYZ data into RPoints
points[i].x = pos.data()[i * W + 0];
points[i].y = pos.data()[i * W + 1];
points[i].z = pos.data()[i * W + 2];
}

cl::KDTree<cl::RPoint3D> kdtree;
kdtree.SwapPoints(&points);
const int k_neighbors = 15;
assert(k_neighbors < N);


//py::print("Running KDTree.");
//cl::Array<cl::RVector3D> normals(N);
cl::Array<cl::Array<int> > neighbors(N);
cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
for (int i = 0; i < N; ++i) {
kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                &neighbors[i]);
  for (int k = 0; k < k_neighbors; ++k) {
neighbor_points[k] = kdtree.points()[neighbors[i][k]];}


}
kdtree.SwapPoints(&points);
cl::Array<int> labels;
cl::Array<VCCSSupervoxel::Supervoxel> supervoxels;
VCCSSupervoxel vccs(points.begin(), points.end(),
                    voxel_resolution,
                    resolution,
                    spatial_importance,
                    normal_importance
                  );
vccs.Segment(&labels, &supervoxels);

// call pure C++ function
//py::gil_scoped_release release;
//py::gil_scoped_acquire acquire;

// TODO: replace this with actual pointcloud colors
cl::Array<cl::RGB32Color> colors = random_colors(points, labels, supervoxels.size());

// todo, we can allocate this explicitely
std::vector<double> result;

// py::print("Copying to output");
for (int i = 0; i < points.size(); i++) {
result.push_back(points[i].x);
result.push_back(points[i].y);
result.push_back(points[i].z);
result.push_back(pos[i * W + 3]);
result.push_back(pos[i * W + 4]);
result.push_back(pos[i * W + 5]);
result.push_back(static_cast<double>(colors[i].red()));
result.push_back(static_cast<double>(colors[i].green()));
result.push_back(static_cast<double>(colors[i].blue()));
// add the label from the segmentation as output..
result.push_back(static_cast<double>(labels[i]));
}

// returns same shape but with one additional point i.e. supervoxel cluster
std::vector<unsigned int> shape   = { N, W + 1};
std::vector<unsigned int> strides = { sizeof(double)*(W + 1) , sizeof(double) };
//py::print("Returning values..");
//py::print(result.size(), shape);

// return 2-D NumPy array
return py::array(py::buffer_info(
result.data(),                           /* data as contiguous array  */
sizeof(double),                          /* size of one scalar        */
py::format_descriptor<double>::format(), /* data type                 */
2,                                       /* number of dimensions      */
shape,                                   /* shape of the matrix       */
strides                                  /* strides for each axis     */
));
}

PYBIND11_MODULE(vccs_supervoxel, m) {
  m.doc() = R"pbdoc(
      Pybind11 VCCS Supervoxel Wrapper
      -----------------------
      .. currentmodule:: vccs_supervoxel
      .. autosummary::
         :toctree: _generate
         VCCSSupervoxel
  )pbdoc";
  m.def("segment", &py_segment, "basic supervoxel clustering using vccs");
  m.def("segment_knn", &py_segment_knn, "basic supervoxel clustering using vccs with knn");
  m.def("segment_vccs", &py_segment_vccs, "basic supervoxel clustering using vccs with vccs");
}
