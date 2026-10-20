#include "BlockFill.hpp"
#include "source/utilities/RandomGenerator.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numbers>

using namespace Mtree;

namespace
{
constexpr int QUADS_PER_BLOCK = 6;

bool is_trunk(const Skeleton& skeleton, const Chain& chain)
{
	return skeleton.segments[(size_t)chain.segments.front()].hierarchy_depth == 0;
}

int blocks_on_chain(const Skeleton& skeleton, const Chain& chain, float density,
                    float trunk_block_length)
{
	if (trunk_block_length > 0.0f && is_trunk(skeleton, chain))
		return std::max(1, (int)std::lround(chain.length / trunk_block_length));
	return std::max(1, (int)std::lround(density * chain.length));
}

int count_blocks(const Skeleton& skeleton, float density, float trunk_block_length)
{
	int blocks = 0;
	for (const Chain& chain : skeleton.chains)
		blocks += blocks_on_chain(skeleton, chain, density, trunk_block_length);
	return blocks;
}

struct BlockSite
{
	const Segment* segment;
	float t;
};

BlockSite resolve_site(const Skeleton& skeleton, const Chain& chain, float distance)
{
	float travelled = 0.0f;
	for (int index : chain.segments)
	{
		const Segment& segment = skeleton.segments[(size_t)index];
		if (segment.length > 0.0f && distance < travelled + segment.length)
			return BlockSite{&segment, (distance - travelled) / segment.length};
		travelled += segment.length;
	}
	const Segment& last = skeleton.segments[(size_t)chain.segments.back()];
	return BlockSite{&last, last.length > 0.0f ? 1.0f : 0.0f};
}

Vector3 axis_from(const Vector3& start, const Vector3& end, const Vector3& fallback)
{
	Vector3 axis = end - start;
	float length = axis.norm();
	return length > 1e-6f ? Vector3(axis / length) : fallback;
}

Vector3 side_axis(const Vector3& z, const Vector3& preferred)
{
	Vector3 x = preferred - z * preferred.dot(z);
	if (x.norm() > 1e-4f)
		return x.normalized();
	for (const Vector3& candidate : {Vector3{1, 0, 0}, Vector3{0, 1, 0}, Vector3{0, 0, 1}})
	{
		Vector3 projected = candidate - z * candidate.dot(z);
		if (projected.norm() > 1e-4f)
			return projected.normalized();
	}
	return preferred;
}

void emit_block(SegmentAttributes& writer, const BlockSite& site, const Vector3& start,
                const Vector3& end, float scale, float roll)
{
	const Segment& segment = *site.segment;
	Vector3 centre = (start + end) / 2.0f;
	Vector3 z = axis_from(start, end, segment.z);
	Vector3 tangent = side_axis(z, segment.x);
	Vector3 side = z.cross(tangent);
	Vector3 x = tangent * std::cos(roll) + side * std::sin(roll);
	Vector3 y = z.cross(x);
	float radius = segment.thickness(site.t);
	float half_width = radius * scale;
	float half_length = (end - start).norm() * scale / 2.0f;

	int corner[2][2][2];
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++)
			{
				Vector3 position = centre + x * (i ? half_width : -half_width) +
				                   y * (j ? half_width : -half_width) +
				                   z * (k ? half_length : -half_length);
				corner[i][j][k] = writer.add_vertex(position, segment, radius, centre);
			}

	auto v = [&](int i, int j, int k) { return corner[i][j][k]; };
	writer.add_unit_quad({v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)}, BARK_MATERIAL);
	writer.add_unit_quad({v(1, 0, 0), v(1, 0, 1), v(1, 1, 1), v(1, 1, 0)}, BARK_MATERIAL);
	writer.add_unit_quad({v(0, 0, 0), v(0, 0, 1), v(1, 0, 1), v(1, 0, 0)}, BARK_MATERIAL);
	writer.add_unit_quad({v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)}, BARK_MATERIAL);
	writer.add_unit_quad({v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)}, BARK_MATERIAL);
	writer.add_unit_quad({v(0, 0, 1), v(0, 1, 1), v(1, 1, 1), v(1, 0, 1)}, BARK_MATERIAL);
}
} // namespace

namespace Mtree
{

int BlockFill::count_polygons(Tree& tree)
{
	block_count = count_blocks(Skeleton::from_tree(tree), density, trunk_block_length);
	polygon_count = block_count * QUADS_PER_BLOCK;
	return polygon_count;
}

Mesh BlockFill::mesh_tree(Tree& tree)
{
	Skeleton skeleton = Skeleton::from_tree(tree);
	block_count = count_blocks(skeleton, density, trunk_block_length);
	polygon_count = block_count * QUADS_PER_BLOCK;

	Mesh mesh;
	SegmentAttributes writer(mesh);

	RandomGenerator random;
	random.set_seed(seed);

	float roll_variation = roll_variation_degrees * std::numbers::pi_v<float> / 180.0f;

	for (const Chain& chain : skeleton.chains)
	{
		int count = blocks_on_chain(skeleton, chain, density, trunk_block_length);
		float step = chain.length / count;
		for (int k = 0; k < count; k++)
		{
			BlockSite base = resolve_site(skeleton, chain, k * step);
			BlockSite tip = resolve_site(skeleton, chain, (k + 1) * step);
			BlockSite middle = resolve_site(skeleton, chain, (k + 0.5f) * step);
			float roll = (2.0f * random.get_0_1() - 1.0f) * roll_variation;
			emit_block(writer, middle, base.segment->axis_point(base.t),
			           tip.segment->axis_point(tip.t), scale, roll);
		}
	}

	return mesh;
}

} // namespace Mtree
