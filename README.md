# Supervoxel for 3D point clouds

A python version of the supervoxel segmentation method for 3D point clouds. The original C++ code can be found [here](https://github.com/yblin/Supervoxel-for-3D-point-clouds) and [here](https://github.com/bastianlb/vccs_supervoxel).

Three supervoxel segmentation method are implemented:
- VCCS: The original VCCS method
- VCCS with kNN: A variation of VCCS that uses kNN to find neighbors.
- yblin's method : [ISPRS 2018 paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271618301370)

Note that there has some problems in the implementation of VCCS with kNN (I will fix as soon as possible). 



## Install & complie

This repository requires C++11 to compile and python3 with pybind11 installed.
Pybin11 can be installed by running `pip install pybind11`.
Then, for python install, clone this repository, change to the root directory and run `pip install .`.

## Sample usage:

Please see the demo.ipynb file for sample usage.

~~~ python
from vccs_supervoxel import segment,segment_knn,segment_vccs
#segment is the main function to call the supervoxel segmentation provided by yblin
#segment_knn is the main function to call the supervoxel segmentation provided by VCCS with kNN
#segment_vccs is the main function to call the supervoxel segmentation provided by VCCS
# Input should have size [N,9] i.e. [N, xyz rgb nxnynz] 
out = segment_vccs(pts[:, :9], 1,1) #or segment_knn(pts[:, :9], 1) or segment(pts[:, :9], 1) 
# out is 2-D NumPy array have size [N,10] containing xyz rgb labelcolor(rgb) label
~~~