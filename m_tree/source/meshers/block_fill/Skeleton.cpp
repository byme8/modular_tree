#include "Skeleton.hpp"
#include "source/utilities/GeometryUtilities.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numbers>

using namespace Mtree;

namespace
{
constexpr float GOLDEN_ANGLE_RAD = 2.39996322972865f;
constexpr float TWO_PI = 2.0f * std::numbers::pi_v<float>;
constexpr float DEGENERATE_TANGENT = 1e-4f;

struct ChainContext
{
	int stem_id;
	int hierarchy_depth;
	Vector3 pivot_position;
	float branch_extent;
};

float calculate_branch_extent(const Node& node)
{
	float extent = node.length;
	if (!node.is_leaf())
		extent += calculate_branch_extent(node.children[0]->node);
	return extent;
}

bool is_degenerate(const Vector3& tangent, const Vector3& direction)
{
	if (!tangent.allFinite())
		return true;
	float norm = tangent.norm();
	if (norm < DEGENERATE_TANGENT)
		return true;
	return std::abs(tangent.dot(direction) / norm) > 1.0f - DEGENERATE_TANGENT;
}

Vector3 resolve_tangent(const Node& node, const Vector3& parent_tangent)
{
	Vector3 direction = node.direction.normalized();
	if (!is_degenerate(node.tangent, direction))
		return Geometry::projected_on_plane(node.tangent, direction).normalized();

	for (const Vector3& candidate :
	     {parent_tangent, Vector3{1, 0, 0}, Vector3{0, 1, 0}, Vector3{0, 0, 1}})
	{
		Vector3 fallback = Geometry::projected_on_plane(candidate, direction);
		if (!is_degenerate(fallback, direction))
			return fallback.normalized();
	}
	return Geometry::get_orthogonal_vector(direction);
}

Segment make_segment(const Node& node, const Vector3& position, const Vector3& parent_tangent,
                     const ChainContext& context, int section_index)
{
	Segment segment;
	segment.start = position;
	segment.z = node.direction.normalized();
	segment.x = resolve_tangent(node, parent_tangent);
	segment.y = segment.z.cross(segment.x).normalized();
	segment.length = node.length;
	segment.radius = node.radius;
	segment.next_radius = node.is_leaf() ? node.radius : node.children[0]->node.radius;
	segment.creator_id = node.creator_id;
	segment.stem_id = context.stem_id;
	segment.hierarchy_depth = context.hierarchy_depth;
	segment.pivot_position = context.pivot_position;
	segment.branch_extent = context.branch_extent;
	segment.phyllotaxis_angle = std::fmod(section_index * GOLDEN_ANGLE_RAD, TWO_PI);
	return segment;
}

Vector3 side_child_position(const Node& parent, const NodeChild& child, const Vector3& position)
{
	Vector3 tangent =
	    Geometry::projected_on_plane(child.node.direction, parent.direction).normalized();
	return position + parent.direction * parent.length * child.position_in_parent +
	       tangent * parent.radius;
}

void collect(const Node& node, const Vector3& position, const Vector3& parent_tangent,
             const ChainContext& context, int section_index, int& stem_id_counter,
             Skeleton& skeleton, Chain& chain)
{
	Segment segment = make_segment(node, position, parent_tangent, context, section_index);
	skeleton.segments.push_back(segment);
	chain.segments.push_back((int)skeleton.segments.size() - 1);
	chain.length += segment.length;
	if (node.is_leaf())
		return;

	Vector3 end = position + node.direction * node.length;
	collect(node.children[0]->node, end, segment.x, context, section_index + 1, stem_id_counter,
	        skeleton, chain);

	for (size_t i = 1; i < node.children.size(); i++)
	{
		const NodeChild& child = *node.children[i];
		Vector3 child_position = side_child_position(node, child, position);
		ChainContext child_context{++stem_id_counter, context.hierarchy_depth + 1,
		                           child_position, calculate_branch_extent(child.node)};
		Chain child_chain{{}, 0.0f};
		collect(child.node, child_position, segment.x, child_context, 0, stem_id_counter,
		        skeleton, child_chain);
		skeleton.chains.push_back(std::move(child_chain));
	}
}

template <typename T> Attribute<T>& ensure_attribute(Mesh& mesh, const std::string& name)
{
	auto found = mesh.attributes.find(name);
	if (found == mesh.attributes.end())
	{
		Attribute<T>& attribute = mesh.add_attribute<T>(name);
		attribute.data.resize(mesh.vertices.size());
		return attribute;
	}
	Attribute<T>& attribute = *static_cast<Attribute<T>*>(found->second.get());
	attribute.data.resize(mesh.vertices.size());
	return attribute;
}

Attribute<Vector3>& ensure_attachment(Mesh& mesh)
{
	bool existed = mesh.attributes.count(SegmentAttributeNames::leaf_attachment) > 0;
	Attribute<Vector3>& attribute =
	    ensure_attribute<Vector3>(mesh, SegmentAttributeNames::leaf_attachment);
	if (!existed)
		for (size_t i = 0; i < mesh.vertices.size(); i++)
			attribute.data[i] = mesh.vertices[i];
	return attribute;
}
} // namespace

namespace Mtree
{

float Segment::thickness(float t) const
{
	return Geometry::lerp(radius, next_radius, std::clamp(t, 0.0f, 1.0f));
}

Vector3 Segment::axis_point(float t) const { return start + z * (t * length); }

Vector3 Segment::frame_point(const Vector3& local) const
{
	return start + x * local[0] + y * local[1] + z * local[2];
}

Skeleton Skeleton::from_tree(Tree& tree)
{
	Skeleton skeleton;
	int stem_id_counter = 0;
	for (Stem& stem : tree.get_stems())
	{
		if (stem.node.children.empty())
			continue;
		ChainContext context{stem_id_counter++, 0, stem.position,
		                     calculate_branch_extent(stem.node)};
		Chain chain{{}, 0.0f};
		collect(stem.node, stem.position, Vector3{1, 0, 0}, context, 0, stem_id_counter,
		        skeleton, chain);
		skeleton.chains.push_back(std::move(chain));
	}
	std::sort(skeleton.chains.begin(), skeleton.chains.end(),
	          [&](const Chain& a, const Chain& b) {
		          return skeleton.segments[(size_t)a.segments[0]].stem_id <
		                 skeleton.segments[(size_t)b.segments[0]].stem_id;
	          });
	return skeleton;
}

int Skeleton::leaf_level() const
{
	int level = 0;
	for (const Segment& segment : segments)
		level = std::max(level, segment.creator_id);
	return level;
}

SegmentAttributes::SegmentAttributes(Mesh& mesh)
    : mesh(mesh), radius(ensure_attribute<float>(mesh, SegmentAttributeNames::radius)),
      direction(ensure_attribute<Vector3>(mesh, SegmentAttributeNames::direction)),
      stem_id(ensure_attribute<float>(mesh, SegmentAttributeNames::stem_id)),
      hierarchy_depth(ensure_attribute<float>(mesh, SegmentAttributeNames::hierarchy_depth)),
      pivot_position(ensure_attribute<Vector3>(mesh, SegmentAttributeNames::pivot_position)),
      branch_extent(ensure_attribute<float>(mesh, SegmentAttributeNames::branch_extent)),
      phyllotaxis_angle(ensure_attribute<float>(mesh, SegmentAttributeNames::phyllotaxis_angle)),
      leaf_attachment(ensure_attachment(mesh))
{
}

int SegmentAttributes::add_vertex(const Vector3& position, const Segment& segment,
                                  float vertex_radius, const Vector3& attachment)
{
	int index = mesh.add_vertex(position);
	radius.data[index] = vertex_radius;
	direction.data[index] = segment.z;
	stem_id.data[index] = (float)segment.stem_id;
	hierarchy_depth.data[index] = (float)segment.hierarchy_depth;
	pivot_position.data[index] = segment.pivot_position;
	branch_extent.data[index] = segment.branch_extent;
	phyllotaxis_angle.data[index] = segment.phyllotaxis_angle;
	leaf_attachment.data[index] = attachment;
	return index;
}

void SegmentAttributes::add_unit_quad(const std::array<int, 4>& corners, int material)
{
	int polygon = mesh.add_polygon();
	mesh.polygons[polygon] = corners;
	int uv = (int)mesh.uvs.size();
	mesh.uvs.emplace_back(0, 0);
	mesh.uvs.emplace_back(1, 0);
	mesh.uvs.emplace_back(1, 1);
	mesh.uvs.emplace_back(0, 1);
	mesh.uv_loops[polygon] = {uv, uv + 1, uv + 2, uv + 3};
	mesh.material_indices[polygon] = material;
}

} // namespace Mtree
