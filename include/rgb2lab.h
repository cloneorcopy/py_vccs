#include <inttypes.h>
#include <stdio.h>
#include <math.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct RGB {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct LAB {
  double l;
  double a;
  double b;
};

double lab_dist(LAB l1, LAB l2) {
  return sqrt(
          pow(l1.l - l2.l, 2) +
          pow(l1.a - l2.a, 2) +
          pow(l1.b - l2.b, 2)
        );
}

double F(double input) // function f(...), which is used for defining L, a and b
                       // changes within [4/29,1]
{
  if (input > 0.008856)
    return std::cbrt(input); // maximum 1 --- prefer cbrt to pow for cubic root
  else
    // return ((double(841) / 108) * input +
    //        double(4) / 29); // 841/108 = 29*29/36*16
    return (float_t(7.787) * input) + (16.0 / 116.0);
}

// RGB to XYZ
void RGBtoXYZ(unsigned char r, unsigned char b, unsigned char g, double &x, double &y, double &z) {
    // Assume RGB has the type invariance satisfied, i.e., channels \in [0,255]
    float_t var_R = float_t(r) / 255;
    float_t var_G = float_t(b) / 255;
    float_t var_B = float_t(g) / 255;

    var_R = (var_R > 0.04045) ? std::pow((var_R + 0.055) / 1.055, 2.4)
                              : var_R / 12.92;
    var_G = (var_G > 0.04045) ? std::pow((var_G + 0.055) / 1.055, 2.4)
                              : var_G / 12.92;
    var_B = (var_B > 0.04045) ? std::pow((var_B + 0.055) / 1.055, 2.4)
                              : var_B / 12.92;

    var_R *= 100;
    var_G *= 100;
    var_B *= 100;

    x = var_R * float_t(0.4124) + var_G * float_t(0.3576) + var_B * float_t(0.1805);
    y = var_R * float_t(0.2126) + var_G * float_t(0.7152) + var_B * float_t(0.0722);
    z = var_R * float_t(0.0193) + var_G * float_t(0.1192) + var_B * float_t(0.9505);
};

// XYZ to CIELab
void XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b)
{
    double Xo = 95.047;
    double Yo = 100;
    double Zo = 108.883;
    L = 116 * F(Y / Yo) - 16; // maximum L = 100
    a = 500 * (F(X / Xo) - F(Y / Yo)); // maximum
    b = 200 * (F(Y / Yo) - F(Z / Zo));
}

// RGB to CIELab
LAB RGB2LAB(RGB rgb)
{
    double X, Y, Z, L, a, b;
    LAB lab;
    RGBtoXYZ(rgb.r, rgb.g, rgb.b, X, Y, Z);
    XYZtoLab(X, Y, Z, L, a, b);
    lab.l = L;
    lab.a = a;
    lab.b = b;
    // py::print(rgb.r, rgb.b, rgb.g, X, Y, Z, L, a, b);
    return lab;
}
