#pragma once
#include "source/mesh/Mesh.hpp"
#include "source/tree/Tree.hpp"
#include <array>
#include <string>
#include <vector>

namespace Mtree
{

struct SegmentAttributeNames
{
	inline static const std::string radius = "radius";
	inline static const std::string direction = "direction";
	inline static const std::string stem_id = "stem_id";
	inline static const std::string hierarchy_depth = "hierarchy_depth";
	inline static const std::string pivot_position = "pivot_position";
	inline static const std::string branch_extent = "branch_extent";
	inline static const std::string phyllotaxis_angle = "phyllotaxis_angle";
	inline static const std::string leaf_attachment = "leaf_attachment";
};

constexpr int BARK_MATERIAL = 0;
constexpr int LEAF_MATERIAL = 1;

struct Segment
{
	Vector3 start;
	Vector3 x;
	Vector3 y;
	Vector3 z;
	float length;
	float radius;
	float next_radius;
	int creator_id;
	int stem_id;
	int hierarchy_depth;
	Vector3 pivot_position;
	float branch_extent;
	float phyllotaxis_angle;

	float thickness(float t) const;
	Vector3 axis_point(float t) const;
	Vector3 frame_point(const Vector3& local) const;
};

Vector3 any_perpendicular(const Vector3& axis, const Vector3& preferred);

struct Chain
{
	std::vector<int> segments;
	float length;
};

struct Skeleton
{
	std::vector<Segment> segments;
	std::vector<Chain> chains;

	static Skeleton from_tree(Tree& tree);
	int leaf_level() const;
};

class SegmentAttributes
{
  public:
	explicit SegmentAttributes(Mesh& mesh);

	int add_vertex(const Vector3& position, const Segment& segment, float radius,
	               const Vector3& attachment);
	void add_unit_quad(const std::array<int, 4>& corners, int material);
	void add_quad(const std::array<int, 4>& corners, const std::array<Vector2, 4>& uvs,
	              int material);

  private:
	Mesh& mesh;
	Attribute<float>& radius;
	Attribute<Vector3>& direction;
	Attribute<float>& stem_id;
	Attribute<float>& hierarchy_depth;
	Attribute<Vector3>& pivot_position;
	Attribute<float>& branch_extent;
	Attribute<float>& phyllotaxis_angle;
	Attribute<Vector3>& leaf_attachment;
};

} // namespace Mtree
