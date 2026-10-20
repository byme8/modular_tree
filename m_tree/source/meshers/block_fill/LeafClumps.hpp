#pragma once
#include "Skeleton.hpp"

namespace Mtree
{

enum class ClumpShape
{
	Bush,
	Cone,
	PineBranch
};

class LeafClumps
{
  public:
	ClumpShape shape = ClumpShape::Bush;
	float voxel_size = 0.05f;
	float clump_size = 0.4f;
	float width = 1.0f;
	float depth = 1.0f;
	float height = 1.0f;
	float size_variation = 0.3f;
	float density = 4.0f;
	float spread = 0.2f;
	float tilt_variation_degrees = 30.0f;
	int tiers = 4;
	float erosion = 0.2f;
	int seed = 0;
	int max_polygons = 200000;

	int clump_count = 0;
	int voxel_count = 0;
	int polygon_count = 0;

	int count_polygons(Tree& tree);
	void append(Tree& tree, Mesh& mesh);
	std::vector<int> append_lod(Tree& tree, Mesh& mesh, float voxel_scale);
};

}
