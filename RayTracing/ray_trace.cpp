#include <algorithm>
#include <bitset>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int WIDTH{10};
constexpr int HEIGHT{10};

struct ColorRGB
{
    int r{}, g{}, b{};
};

struct Vec3d
{
    double x{}, y{}, z{};

    double norm()
    {
        return sqrt(x * x + y * y + z * z);
    }
};

Vec3d operator-(const Vec3d &lhs, const Vec3d &rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3d operator+(const Vec3d &lhs, const Vec3d &rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vec3d operator*(const Vec3d &vec, double scalar)
{
    return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

Vec3d operator/(const Vec3d &vec, double scalar)
{
    return {vec.x / scalar, vec.y / scalar, vec.z / scalar};
}

Vec3d operator-(const Vec3d &vec)
{
    return {-vec.x, -vec.y, -vec.z};
}

struct HalfSpace
{
    Vec3d    p{};
    Vec3d    n{};
    ColorRGB color{};
};

double dot(const Vec3d &p1, const Vec3d &p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

struct Ray
{
    Vec3d origin{}; // ray origin
    Vec3d dir{};    // normalized ray direction from origin
};

void readHalfSpaces(std::vector<HalfSpace> &half_spaces_out)
{
    half_spaces_out.clear();
    std::string line;
    std::getline(std::cin, line); // read the header line

    // Parse the header line to determine the order of the columns
    std::stringstream        header_ss(line);
    std::string              header;
    std::vector<std::string> headers;
    while (header_ss >> header)
    {
        headers.push_back(header);
    }

    // Read each data line
    while (std::getline(std::cin, line))
    {
        std::stringstream ss(line);
        std::vector<int>  values(9); // to hold px, py, pz, nx, ny, nz, r, g, b in order
        int               value;
        int               index = 0;

        while (ss >> value)
        {
            if (headers[index] == "px")
                values[0] = value;
            else if (headers[index] == "py")
                values[1] = value;
            else if (headers[index] == "pz")
                values[2] = value;
            else if (headers[index] == "nx")
                values[3] = value;
            else if (headers[index] == "ny")
                values[4] = value;
            else if (headers[index] == "nz")
                values[5] = value;
            else if (headers[index] == "r")
                values[6] = value;
            else if (headers[index] == "g")
                values[7] = value;
            else if (headers[index] == "b")
                values[8] = value;
            index++;
        }

        half_spaces_out.push_back(
            HalfSpace{{static_cast<double>(values[0]), static_cast<double>(values[1]), static_cast<double>(values[2])},
                      {static_cast<double>(values[3]), static_cast<double>(values[4]), static_cast<double>(values[5])},
                      ColorRGB{values[6], values[7], values[8]}});
    }
}

// Reference for ray plane intersection equation:
// http://lousodrome.net/blog/light/2020/07/03/intersection-of-a-ray-and-a-plane/
double rayHalfSpaceIntersection(const HalfSpace &half_space, const Ray &ray)
{
    double denominator = dot(ray.dir, half_space.n);

    // If the ray is parallel to the plane, there is no intersection
    constexpr double kDoubleEps{1e-9};
    if (std::abs(denominator) < kDoubleEps)
        return -1.0;

    // Calculate t = distance along the ray
    double t = dot(half_space.p - ray.origin, half_space.n) / denominator;

    // If t is negative, the intersection point is behind the camera
    if (t < 0)
        return -1.0;

    return t;
}

ColorRGB rayCasting(const Ray &ray, const std::vector<HalfSpace> &half_spaces)
{
    ColorRGB color_out    = {0, 0, 0};
    double   closest_dist = std::numeric_limits<double>::infinity();

    // Loop through all half-spaces
    for (const auto &hs : half_spaces)
    {
        // Calculate intersection point (distance along ray direction)
        double t = rayHalfSpaceIntersection(hs, ray);

        // Check if ray hits the half-space plane and the hit is closer than the current closest hit
        if (t >= 0.0 && t < closest_dist)
        {
            // Check if the intersection point is inside all the other half-spaces
            // dot((x,y,z)-(px,py,pz), (nx,ny,nz)) < 0
            bool inside_all = true;
            for (const auto &other_hs : half_spaces)
            {
                if (&hs == &other_hs)
                    continue;

                Vec3d ray_hs_intersection_pt{ray.origin + ray.dir * t};
                if (dot(ray_hs_intersection_pt - other_hs.p, other_hs.n) > 0)
                {
                    inside_all = false;
                    break;
                }
            }
            if (inside_all)
            {
                color_out    = hs.color;
                closest_dist = t;
            }
        }
    }
    return color_out;
}

int main()
{
    std::vector<HalfSpace> half_spaces;
    readHalfSpaces(half_spaces);

    // std::ofstream out("output.ppm");
    std::ostream &out = std::cout;
    out << "P3\n" << WIDTH << ' ' << HEIGHT << "\n9\n";

    // Cast a ray for each pixel, originating from camera center(0,0,0)
    for (int v = 0; v < HEIGHT; ++v)
    {
        for (int u = 0; u < WIDTH; ++u)
        {
            Vec3d ray_dir{u - 0.5 * static_cast<double>(WIDTH) + 0.5,
                          v - 0.5 * static_cast<double>(HEIGHT) + 0.5,
                          0.5 * static_cast<double>(HEIGHT)};
            Ray   ray{{0, 0, 0}, {ray_dir / ray_dir.norm()}};

            ColorRGB pixelColor{rayCasting(ray, half_spaces)};

            out << pixelColor.r << ' ' << pixelColor.g << ' ' << pixelColor.b << ' ';
        }
        out << '\n';
    }

    // out.close();

    return 0;
}