#include "LeafCards.hpp"
#include "source/utilities/RandomGenerator.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <numbers>

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

Vector3 any_perpendicular(const Vector3& axis, const Vector3& preferred)
{
	Vector3 candidate = preferred - axis * preferred.dot(axis);
	if (candidate.norm() > 1e-4f)
		return candidate.normalized();
	for (const Vector3& fallback : {Vector3{1, 0, 0}, Vector3{0, 1, 0}, Vector3{0, 0, 1}})
	{
		candidate = fallback - axis * fallback.dot(axis);
		if (candidate.norm() > 1e-4f)
			return candidate.normalized();
	}
	return preferred;
}

void emit_cards(SegmentAttributes& writer, const Segment& segment, int count, float card_size,
                const CardPlacement& placement, RandomGenerator& random)
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

		Vector3 a_side = tangential * half;
		int a0 = writer.add_vertex(centre - a_side - up, segment, radius, attachment);
		int a1 = writer.add_vertex(centre - a_side + up, segment, radius, attachment);
		int a2 = writer.add_vertex(centre + a_side + up, segment, radius, attachment);
		int a3 = writer.add_vertex(centre + a_side - up, segment, radius, attachment);
		writer.add_unit_quad({a0, a1, a2, a3}, LEAF_MATERIAL);

		Vector3 b_side = radial * half;
		int b0 = writer.add_vertex(centre - b_side - up, segment, radius, attachment);
		int b1 = writer.add_vertex(centre + b_side - up, segment, radius, attachment);
		int b2 = writer.add_vertex(centre + b_side + up, segment, radius, attachment);
		int b3 = writer.add_vertex(centre - b_side + up, segment, radius, attachment);
		writer.add_unit_quad({b0, b1, b2, b3}, LEAF_MATERIAL);
	}
}
} // namespace

namespace Mtree
{

int LeafCards::count_polygons(Tree& tree)
{
	card_count = count_cards(Skeleton::from_tree(tree), card_size, density);
	return card_count * QUADS_PER_CARD;
}

void LeafCards::append(Tree& tree, Mesh& mesh)
{
	Skeleton skeleton = Skeleton::from_tree(tree);
	card_count = count_cards(skeleton, card_size, density);
	if (card_count == 0)
		return;

	int level = skeleton.leaf_level();
	SegmentAttributes writer(mesh);

	RandomGenerator random;
	random.set_seed(seed);

	CardPlacement placement{spread, tilt_variation_degrees * std::numbers::pi_v<float> / 180.0f};

	for (const Segment& segment : skeleton.segments)
		if (segment.creator_id == level)
			emit_cards(writer, segment, cards_on_segment(segment, card_size, density), card_size,
			           placement, random);
}

} // namespace Mtree
