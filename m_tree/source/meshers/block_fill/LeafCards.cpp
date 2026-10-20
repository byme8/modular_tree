#include "LeafCards.hpp"
#include "source/utilities/RandomGenerator.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <vector>

using namespace Mtree;

namespace
{
constexpr float TWO_PI = 2.0f * std::numbers::pi_v<float>;
constexpr int QUADS_PER_CARD = 2;

int cards_on_segment(const Segment& segment, float card_size, float density)
{
	if (card_size <= 0.0f || density <= 0.0f)
		return 0;
	return (int)std::lround(density * segment.length);
}

int count_cards(const Skeleton& skeleton, float card_size, float density)
{
	int level = skeleton.leaf_level();
	int cards = 0;
	for (const Segment& segment : skeleton.segments)
		if (segment.creator_id == level)
			cards += cards_on_segment(segment, card_size, density);
	return cards;
}

struct CardPlacement
{
	float spread;
	float tilt_variation;
};

struct Card
{
	int segment;
	float radius;
	Vector3 attachment;
	Vector3 centre;
	Vector3 up;
	Vector3 a_side;
	Vector3 b_side;
};

void place_segment_cards(std::vector<Card>& cards, const Segment& segment, int segment_index,
                         int count, float card_size, const CardPlacement& placement,
                         RandomGenerator& random)
{
	float half = card_size / 2.0f;
	for (int card = 0; card < count; card++)
	{
		float t = random.get_0_1();
		float phi = random.get_0_1() * TWO_PI;
		float reach = std::sqrt(random.get_0_1()) * placement.spread;
		float roll = random.get_0_1() * TWO_PI;
		float tilt = random.get_0_1() * placement.tilt_variation;
		float azimuth = random.get_0_1() * TWO_PI;

		float radius = segment.thickness(t);
		Vector3 attachment = segment.axis_point(t);
		Vector3 radial_phi = segment.x * std::cos(phi) + segment.y * std::sin(phi);
		Vector3 centre = attachment + radial_phi * (radius + reach);

		Vector3 lean = segment.x * std::cos(azimuth) + segment.y * std::sin(azimuth);
		Vector3 axis = (segment.z * std::cos(tilt) + lean * std::sin(tilt)).normalized();
		Vector3 first = any_perpendicular(axis, segment.x);
		Vector3 second = axis.cross(first);

		Vector3 radial = first * std::cos(roll) + second * std::sin(roll);
		Vector3 tangential = -first * std::sin(roll) + second * std::cos(roll);
		Vector3 up = axis * half;

		cards.push_back(
		    Card{segment_index, radius, attachment, centre, up, tangential * half, radial * half});
	}
}

std::vector<Card> place_cards(const Skeleton& skeleton, float card_size, float density,
                              const CardPlacement& placement, int seed)
{
	std::vector<Card> cards;
	if (count_cards(skeleton, card_size, density) == 0)
		return cards;

	int level = skeleton.leaf_level();
	RandomGenerator random;
	random.set_seed(seed);

	for (int index = 0; index < (int)skeleton.segments.size(); index++)
	{
		const Segment& segment = skeleton.segments[(size_t)index];
		if (segment.creator_id == level)
			place_segment_cards(cards, segment, index,
			                    cards_on_segment(segment, card_size, density), card_size, placement,
			                    random);
	}
	return cards;
}

void emit_card(SegmentAttributes& writer, const Segment& segment, const Card& card)
{
	const Vector3& centre = card.centre;
	const Vector3& up = card.up;
	const Vector3& a_side = card.a_side;
	const Vector3& b_side = card.b_side;
	float radius = card.radius;
	const Vector3& attachment = card.attachment;

	int a0 = writer.add_vertex(centre - a_side - up, segment, radius, attachment);
	int a1 = writer.add_vertex(centre - a_side + up, segment, radius, attachment);
	int a2 = writer.add_vertex(centre + a_side + up, segment, radius, attachment);
	int a3 = writer.add_vertex(centre + a_side - up, segment, radius, attachment);
	writer.add_unit_quad({a0, a1, a2, a3}, LEAF_MATERIAL);

	int b0 = writer.add_vertex(centre - b_side - up, segment, radius, attachment);
	int b1 = writer.add_vertex(centre + b_side - up, segment, radius, attachment);
	int b2 = writer.add_vertex(centre + b_side + up, segment, radius, attachment);
	int b3 = writer.add_vertex(centre - b_side + up, segment, radius, attachment);
	writer.add_unit_quad({b0, b1, b2, b3}, LEAF_MATERIAL);
}

Card enlarged(Card card, float scale)
{
	card.up *= scale;
	card.a_side *= scale;
	card.b_side *= scale;
	return card;
}

constexpr int EXPOSURE_DIRECTIONS = 32;
constexpr int EXPOSURE_GRID_CELLS = 32;
constexpr double LEAF_PROJECTION = 0.5;

std::vector<Vector3> sphere_directions(int count)
{
	const float golden_angle = std::numbers::pi_v<float> * (3.0f - std::sqrt(5.0f));
	std::vector<Vector3> directions;
	for (int i = 0; i < count; i++)
	{
		float z = 1.0f - (2.0f * (float)i + 1.0f) / (float)count;
		float ring = std::sqrt(std::max(0.0f, 1.0f - z * z));
		float angle = golden_angle * (float)i;
		directions.emplace_back(ring * std::cos(angle), ring * std::sin(angle), z);
	}
	return directions;
}

class LeafAreaGrid
{
  public:
	LeafAreaGrid(const std::vector<Vector3>& centres, const std::vector<double>& areas, float margin)
	{
		Eigen::AlignedBox3f bounds;
		for (const Vector3& centre : centres)
			bounds.extend(centre);
		bounds.min().array() -= margin;
		bounds.max().array() += margin;

		origin = bounds.min();
		cell = std::max(bounds.sizes().maxCoeff() / (float)EXPOSURE_GRID_CELLS, 1e-4f);
		for (int axis = 0; axis < 3; axis++)
			dims[axis] = std::max(1, (int)std::ceil(bounds.sizes()[axis] / cell));
		extent = origin + Vector3((float)dims[0], (float)dims[1], (float)dims[2]) * cell;
		area_density.assign((size_t)dims[0] * dims[1] * dims[2], 0.0);

		double cell_volume = (double)cell * cell * cell;
		for (size_t i = 0; i < centres.size(); i++)
			area_density[(size_t)index_of(centres[i])] += areas[i] / cell_volume;
	}

	double transmittance(const Vector3& start, const Vector3& direction) const
	{
		float step = cell * 0.5f;
		float entry = 0.0f;
		bool outside = index_of(start) < 0;
		if (outside && !ray_enters(start, direction, entry))
			return 1.0;

		double optical_depth = 0.0;
		for (float distance = step;; distance += step)
		{
			int index = index_of(start + direction * distance);
			if (index < 0)
			{
				if (outside && distance < entry + step)
					continue;
				break;
			}
			optical_depth += area_density[(size_t)index] * (double)step;
		}
		return std::exp(-LEAF_PROJECTION * optical_depth);
	}

  private:
	Vector3 origin;
	Vector3 extent;
	float cell = 1.0f;
	int dims[3] = {1, 1, 1};
	std::vector<double> area_density;

	bool ray_enters(const Vector3& start, const Vector3& direction, float& entry) const
	{
		float entering = 0.0f;
		float leaving = std::numeric_limits<float>::max();
		for (int axis = 0; axis < 3; axis++)
		{
			if (std::abs(direction[axis]) < 1e-8f)
			{
				if (start[axis] < origin[axis] || start[axis] > extent[axis])
					return false;
				continue;
			}
			float a = (origin[axis] - start[axis]) / direction[axis];
			float b = (extent[axis] - start[axis]) / direction[axis];
			entering = std::max(entering, std::min(a, b));
			leaving = std::min(leaving, std::max(a, b));
		}
		entry = entering;
		return entering <= leaving;
	}

	int index_of(const Vector3& point) const
	{
		int coordinates[3];
		for (int axis = 0; axis < 3; axis++)
		{
			int coordinate = (int)std::floor((point[axis] - origin[axis]) / cell);
			if (coordinate < 0 || coordinate >= dims[axis])
				return -1;
			coordinates[axis] = coordinate;
		}
		return (coordinates[2] * dims[1] + coordinates[1]) * dims[0] + coordinates[0];
	}
};

std::vector<float> exposure_at(const LeafAreaGrid& grid, const std::vector<Vector3>& points)
{
	std::vector<Vector3> directions = sphere_directions(EXPOSURE_DIRECTIONS);
	std::vector<float> exposure;
	exposure.reserve(points.size());
	for (const Vector3& point : points)
	{
		double sum = 0.0;
		for (const Vector3& direction : directions)
			sum += grid.transmittance(point, direction);
		exposure.push_back((float)(sum / (double)directions.size()));
	}
	return exposure;
}

std::vector<float> exposure_of(const std::vector<Card>& cards, float card_size)
{
	if (cards.empty())
		return {};

	std::vector<Vector3> centres;
	for (const Card& card : cards)
		centres.push_back(card.centre);
	std::vector<double> areas(cards.size(), 2.0 * (double)card_size * (double)card_size);
	return exposure_at(LeafAreaGrid(centres, areas, card_size), centres);
}

std::vector<int> most_exposed(const std::vector<Card>& cards, float card_size, float keep_fraction)
{
	int total = (int)cards.size();
	int keep = std::clamp((int)std::lround(keep_fraction * (float)total), 0, total);

	std::vector<float> exposure = exposure_of(cards, card_size);
	std::vector<int> order(cards.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(),
	                 [&](int a, int b) { return exposure[(size_t)a] > exposure[(size_t)b]; });

	std::vector<int> kept(order.begin(), order.begin() + keep);
	std::sort(kept.begin(), kept.end());
	return kept;
}
} // namespace

namespace Mtree
{

int LeafCards::count_polygons(Tree& tree)
{
	card_count = count_cards(Skeleton::from_tree(tree), card_size, density);
	return card_count * QUADS_PER_CARD;
}

static CardPlacement placement_of(const LeafCards& settings)
{
	return CardPlacement{settings.spread,
	                     settings.tilt_variation_degrees * std::numbers::pi_v<float> / 180.0f};
}

void LeafCards::append(Tree& tree, Mesh& mesh, float keep_fraction)
{
	Skeleton skeleton = Skeleton::from_tree(tree);
	std::vector<Card> cards = place_cards(skeleton, card_size, density, placement_of(*this), seed);
	card_count = (int)cards.size();
	if (card_count == 0)
		return;

	if (keep_fraction >= 1.0f)
	{
		SegmentAttributes writer(mesh);
		for (const Card& card : cards)
			emit_card(writer, skeleton.segments[(size_t)card.segment], card);
		return;
	}

	std::vector<int> kept = most_exposed(cards, card_size, keep_fraction);
	card_count = (int)kept.size();
	if (kept.empty())
		return;

	float enlargement = std::sqrt((float)cards.size() / (float)kept.size());
	SegmentAttributes writer(mesh);
	for (int index : kept)
	{
		const Card& card = cards[(size_t)index];
		emit_card(writer, skeleton.segments[(size_t)card.segment], enlarged(card, enlargement));
	}
}

std::vector<float> LeafCards::card_exposure(Tree& tree)
{
	std::vector<Card> cards =
	    place_cards(Skeleton::from_tree(tree), card_size, density, placement_of(*this), seed);
	return exposure_of(cards, card_size);
}

std::vector<float> sky_exposure(const std::vector<Vector3>& occluder_centres,
                                const std::vector<double>& occluder_areas, float margin,
                                const std::vector<Vector3>& points)
{
	if (occluder_centres.empty())
		return std::vector<float>(points.size(), 1.0f);
	return exposure_at(LeafAreaGrid(occluder_centres, occluder_areas, margin), points);
}

std::vector<int> LeafCards::lod_cards(Tree& tree, float keep_fraction)
{
	std::vector<Card> cards =
	    place_cards(Skeleton::from_tree(tree), card_size, density, placement_of(*this), seed);
	if (keep_fraction >= 1.0f)
	{
		std::vector<int> all(cards.size());
		std::iota(all.begin(), all.end(), 0);
		return all;
	}
	return most_exposed(cards, card_size, keep_fraction);
}

} // namespace Mtree
