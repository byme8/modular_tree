#pragma once
#include "Skeleton.hpp"

namespace Mtree
{

class LeafCards
{
  public:
	float card_size = 0.2f;
	float density = 20.0f;
	float spread = 0.3f;
	float tilt_variation_degrees = 90.0f;
	int seed = 0;

	int card_count = 0;

	int count_polygons(Tree& tree);
	void append(Tree& tree, Mesh& mesh);
};

} // namespace Mtree
