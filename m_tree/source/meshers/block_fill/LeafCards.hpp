#pragma once
#include "Skeleton.hpp"
#include <vector>

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
	void append(Tree& tree, Mesh& mesh, float keep_fraction = 1.0f);

	std::vector<float> card_exposure(Tree& tree);
	std::vector<int> lod_cards(Tree& tree, float keep_fraction);
};

std::vector<float> sky_exposure(const std::vector<Vector3>& occluder_centres,
                                const std::vector<double>& occluder_areas, float margin,
                                const std::vector<Vector3>& points);

} // namespace Mtree
