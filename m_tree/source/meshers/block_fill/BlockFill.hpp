#pragma once
#include "Skeleton.hpp"

namespace Mtree
{

class BlockFill
{
  public:
	float density = 10.0f;
	float scale = 1.0f;
	float trunk_block_length = 0.0f;
	float roll_variation_degrees = 15.0f;
	int seed = 0;

	int block_count = 0;
	int polygon_count = 0;

	int count_polygons(Tree& tree);
	Mesh mesh_tree(Tree& tree);
};

} // namespace Mtree
