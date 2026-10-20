#include "LeafClumps.hpp"
#include "source/utilities/RandomGenerator.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace Mtree;

namespace
{
constexpr float TWO_PI = 2.0f * std::numbers::pi_v<float>;
constexpr double PI = std::numbers::pi;
constexpr double ESTIMATE_POLYGON_MULTIPLE = 10.0;

using Matrix3 = Eigen::Matrix3f;
using Cell = std::array<int, 3>;

uint64_t mix(uint64_t value)
{
	value += 0x9E3779B97F4A7C15ull;
	value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
	value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
	return value ^ (value >> 31);
}

uint64_t hash_cell(const Cell& cell, uint64_t salt)
{
	uint64_t hash = mix(salt);
	for (int coordinate : cell)
		hash = mix(hash ^ (uint64_t)(uint32_t)coordinate);
	return hash;
}

double unit_hash(const Cell& cell, int seed)
{
	return (double)(hash_cell(cell, (uint64_t)(uint32_t)seed) >> 11) * 0x1.0p-53;
}

struct CellHash
{
	size_t operator()(const Cell& cell) const { return (size_t)hash_cell(cell, 0); }
};

struct Settings
{
	ClumpShape shape;
	float voxel;
	float size;
	float width;
	float depth;
	float height;
	float size_variation;
	float density;
	float spread;
	float tilt_variation;
	int tiers;
	float erosion;
	int seed;
	int max_polygons;
};

Settings sanitized(const LeafClumps& clumps)
{
	return Settings{clumps.shape,
	                std::clamp(clumps.voxel_size, 0.005f, 1.0f),
	                std::clamp(clumps.clump_size, 0.0f, 10.0f),
	                std::clamp(clumps.width, 0.1f, 10.0f),
	                std::clamp(clumps.depth, 0.1f, 10.0f),
	                std::clamp(clumps.height, 0.1f, 10.0f),
	                std::clamp(clumps.size_variation, 0.0f, 0.9f),
	                std::clamp(clumps.density, 0.0f, 1000.0f),
	                std::clamp(clumps.spread, 0.0f, 10.0f),
	                std::clamp(clumps.tilt_variation_degrees, 0.0f, 90.0f) *
	                    std::numbers::pi_v<float> / 180.0f,
	                std::clamp(clumps.tiers, 1, 16),
	                std::clamp(clumps.erosion, 0.0f, 0.9f),
	                clumps.seed,
	                std::clamp(clumps.max_polygons, 1000, 10000000)};
}

struct Clump
{
	int segment;
	float radius;
	Vector3 attachment;
	Vector3 centre;
	Matrix3 frame;
	Vector3 half;
};

int clumps_on_segment(const Segment& segment, const Settings& settings)
{
	if (settings.size <= 0.0f || settings.density <= 0.0f)
		return 0;
	return (int)std::lround(settings.density * segment.length);
}

Matrix3 pine_frame(const Segment& segment, float tilt, float azimuth)
{
	Vector3 lean = segment.x * std::cos(azimuth) + segment.y * std::sin(azimuth);
	Vector3 axis = (segment.z * std::cos(tilt) + lean * std::sin(tilt)).normalized();
	Vector3 first = any_perpendicular(axis, segment.x);
	Matrix3 frame;
	frame.col(0) = first;
	frame.col(1) = axis.cross(first);
	frame.col(2) = axis;
	return frame;
}

std::vector<Clump> place_clumps(const Skeleton& skeleton, const Settings& settings)
{
	std::vector<Clump> clumps;
	int level = skeleton.leaf_level();
	RandomGenerator random;
	random.set_seed(settings.seed);

	for (int index = 0; index < (int)skeleton.segments.size(); index++)
	{
		const Segment& segment = skeleton.segments[(size_t)index];
		if (segment.creator_id != level)
			continue;
		int count = clumps_on_segment(segment, settings);
		for (int clump = 0; clump < count; clump++)
		{
			float t = random.get_0_1();
			float phi = random.get_0_1() * TWO_PI;
			float reach = std::sqrt(random.get_0_1()) * settings.spread;
			float size_factor = 1.0f + (2.0f * random.get_0_1() - 1.0f) * settings.size_variation;
			float tilt = random.get_0_1() * settings.tilt_variation;
			float azimuth = random.get_0_1() * TWO_PI;

			float radius = segment.thickness(t);
			Vector3 attachment = segment.axis_point(t);
			Vector3 radial = segment.x * std::cos(phi) + segment.y * std::sin(phi);
			Vector3 centre = attachment + radial * (radius + reach);

			float size = settings.size * size_factor;
			Vector3 half{size * settings.width / 2.0f, size * settings.depth / 2.0f,
			             size * settings.height / 2.0f};
			Matrix3 frame = settings.shape == ClumpShape::PineBranch
			                    ? pine_frame(segment, tilt, azimuth)
			                    : Matrix3::Identity();
			clumps.push_back(Clump{index, radius, attachment, centre, frame, half});
		}
	}
	return clumps;
}

float tier_spacing(const Clump& clump, const Settings& settings)
{
	return 2.0f * clump.half.z() / (float)settings.tiers;
}

float tier_half_thickness(const Clump& clump, const Settings& settings)
{
	return std::max(tier_spacing(clump, settings) / 2.0f, settings.voxel) / 2.0f;
}

float tier_scale(int tier, const Settings& settings)
{
	return 1.0f - (float)tier / (float)settings.tiers;
}

bool inside(const Clump& clump, const Settings& settings, const Vector3& local)
{
	const Vector3& half = clump.half;
	switch (settings.shape)
	{
	case ClumpShape::Bush:
		return local.cwiseQuotient(half).squaredNorm() <= 1.0f;
	case ClumpShape::Cone:
	{
		float rise = (local.z() + half.z()) / (2.0f * half.z());
		if (rise < 0.0f || rise > 1.0f)
			return false;
		float across = local.head<2>().cwiseQuotient(half.head<2>()).squaredNorm();
		return across <= (1.0f - rise) * (1.0f - rise);
	}
	case ClumpShape::PineBranch:
	{
		float spacing = tier_spacing(clump, settings);
		float half_thickness = tier_half_thickness(clump, settings);
		for (int tier = 0; tier < settings.tiers; tier++)
		{
			float tier_centre = -half.z() + ((float)tier + 0.5f) * spacing;
			if (std::abs(local.z() - tier_centre) > half_thickness)
				continue;
			Vector2 radii = half.head<2>() * tier_scale(tier, settings);
			if (local.head<2>().cwiseQuotient(radii).squaredNorm() <= 1.0f)
				return true;
		}
		return false;
	}
	}
	return false;
}

Vector3 local_bounds(const Clump& clump, const Settings& settings)
{
	Vector3 bounds = clump.half;
	if (settings.shape == ClumpShape::PineBranch)
		bounds.z() += tier_half_thickness(clump, settings);
	return bounds;
}

double solid_volume(const Clump& clump, const Settings& settings)
{
	double area = PI * (double)clump.half.x() * (double)clump.half.y();
	switch (settings.shape)
	{
	case ClumpShape::Bush:
		return 4.0 / 3.0 * area * (double)clump.half.z();
	case ClumpShape::Cone:
		return area * 2.0 * (double)clump.half.z() / 3.0;
	case ClumpShape::PineBranch:
	{
		double thickness = 2.0 * (double)tier_half_thickness(clump, settings);
		double volume = 0.0;
		for (int tier = 0; tier < settings.tiers; tier++)
		{
			double scale = (double)tier_scale(tier, settings);
			volume += area * scale * scale * thickness;
		}
		return volume;
	}
	}
	return 0.0;
}

struct Occupant
{
	int owner;
	float distance;
};

using VoxelMap = std::unordered_map<Cell, Occupant, CellHash>;

VoxelMap fill(const std::vector<Clump>& clumps, const Settings& settings)
{
	VoxelMap voxels;
	float voxel = settings.voxel;
	for (int owner = 0; owner < (int)clumps.size(); owner++)
	{
		const Clump& clump = clumps[(size_t)owner];
		Vector3 reach = clump.frame.cwiseAbs() * local_bounds(clump, settings);
		Cell low;
		Cell high;
		for (int axis = 0; axis < 3; axis++)
		{
			low[axis] = (int)std::ceil((clump.centre[axis] - reach[axis]) / voxel - 0.5f);
			high[axis] = (int)std::floor((clump.centre[axis] + reach[axis]) / voxel - 0.5f);
		}
		for (int z = low[2]; z <= high[2]; z++)
			for (int y = low[1]; y <= high[1]; y++)
				for (int x = low[0]; x <= high[0]; x++)
				{
					Vector3 point = (Vector3((float)x, (float)y, (float)z).array() + 0.5f) * voxel;
					Vector3 local = clump.frame.transpose() * (point - clump.centre);
					if (!inside(clump, settings, local))
						continue;
					float distance = local.cwiseQuotient(clump.half).norm();
					auto [found, inserted] = voxels.try_emplace(Cell{x, y, z}, Occupant{owner, distance});
					if (!inserted && distance < found->second.distance)
						found->second = Occupant{owner, distance};
				}
	}
	return voxels;
}

Cell step(const Cell& cell, int axis, int sign)
{
	Cell next = cell;
	next[(size_t)axis] += sign;
	return next;
}

bool same_owner(const VoxelMap& voxels, const Cell& cell, int owner)
{
	auto found = voxels.find(cell);
	return found != voxels.end() && found->second.owner == owner;
}

bool on_surface(const VoxelMap& voxels, const Cell& cell, int owner)
{
	for (int axis = 0; axis < 3; axis++)
		for (int sign : {1, -1})
			if (!same_owner(voxels, step(cell, axis, sign), owner))
				return true;
	return false;
}

void erode(VoxelMap& voxels, const Settings& settings)
{
	if (settings.erosion <= 0.0f)
		return;
	std::vector<Cell> removed;
	for (const auto& [cell, occupant] : voxels)
		if (on_surface(voxels, cell, occupant.owner) &&
		    unit_hash(cell, settings.seed) < (double)settings.erosion)
			removed.push_back(cell);
	for (const Cell& cell : removed)
		voxels.erase(cell);
}

struct OwnedCell
{
	int owner;
	Cell cell;
};

std::vector<OwnedCell> by_owner(const VoxelMap& voxels)
{
	std::vector<OwnedCell> cells;
	cells.reserve(voxels.size());
	for (const auto& [cell, occupant] : voxels)
		cells.push_back(OwnedCell{occupant.owner, cell});
	std::sort(cells.begin(), cells.end(), [](const OwnedCell& a, const OwnedCell& b) {
		if (a.owner != b.owner)
			return a.owner < b.owner;
		return std::tie(a.cell[2], a.cell[1], a.cell[0]) < std::tie(b.cell[2], b.cell[1], b.cell[0]);
	});
	return cells;
}

struct Quad
{
	int owner;
	std::array<Cell, 4> corners;
	std::array<Vector2, 4> uvs;
};

struct Face
{
	int slice;
	int u;
	int w;

	bool operator<(const Face& other) const
	{
		return std::tie(slice, w, u) < std::tie(other.slice, other.w, other.u);
	}
};

uint64_t face_key(int u, int w) { return ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)w; }

Quad face_quad(int owner, int axis, int sign, int slice, int u, int w, int du, int dw)
{
	int u_axis = (axis + 1) % 3;
	int w_axis = (axis + 2) % 3;
	std::array<std::array<int, 2>, 4> offsets =
	    sign > 0 ? std::array<std::array<int, 2>, 4>{{{0, 0}, {0, dw}, {du, dw}, {du, 0}}}
	             : std::array<std::array<int, 2>, 4>{{{0, 0}, {du, 0}, {du, dw}, {0, dw}}};
	Quad quad{owner, {}, {}};
	for (size_t corner = 0; corner < 4; corner++)
	{
		Cell& position = quad.corners[corner];
		position[(size_t)axis] = slice;
		position[(size_t)u_axis] = u + offsets[corner][0];
		position[(size_t)w_axis] = w + offsets[corner][1];
		quad.uvs[corner] = Vector2((float)offsets[corner][0], (float)offsets[corner][1]);
	}
	return quad;
}

void merge_slice(std::vector<Quad>& quads, const std::vector<Face>& faces, size_t begin, size_t end,
                 int owner, int axis, int sign)
{
	std::unordered_map<uint64_t, bool> consumed;
	for (size_t i = begin; i < end; i++)
		consumed[face_key(faces[i].u, faces[i].w)] = false;
	auto open = [&](int u, int w) {
		auto found = consumed.find(face_key(u, w));
		return found != consumed.end() && !found->second;
	};

	for (size_t i = begin; i < end; i++)
	{
		const Face& face = faces[i];
		if (!open(face.u, face.w))
			continue;
		int du = 1;
		while (open(face.u + du, face.w))
			du++;
		int dw = 1;
		for (;; dw++)
		{
			bool row_open = true;
			for (int k = 0; k < du && row_open; k++)
				row_open = open(face.u + k, face.w + dw);
			if (!row_open)
				break;
		}
		for (int j = 0; j < dw; j++)
			for (int k = 0; k < du; k++)
				consumed[face_key(face.u + k, face.w + j)] = true;
		quads.push_back(face_quad(owner, axis, sign, face.slice, face.u, face.w, du, dw));
	}
}

void mesh_owner(std::vector<Quad>& quads, const VoxelMap& voxels, const std::vector<OwnedCell>& cells,
                size_t begin, size_t end)
{
	int owner = cells[begin].owner;
	for (int axis = 0; axis < 3; axis++)
		for (int sign : {1, -1})
		{
			std::vector<Face> faces;
			for (size_t i = begin; i < end; i++)
			{
				const Cell& cell = cells[i].cell;
				if (same_owner(voxels, step(cell, axis, sign), owner))
					continue;
				faces.push_back(Face{cell[(size_t)axis] + (sign > 0 ? 1 : 0),
				                     cell[(size_t)((axis + 1) % 3)], cell[(size_t)((axis + 2) % 3)]});
			}
			std::sort(faces.begin(), faces.end());
			for (size_t first = 0; first < faces.size();)
			{
				size_t last = first;
				while (last < faces.size() && faces[last].slice == faces[first].slice)
					last++;
				merge_slice(quads, faces, first, last, owner, axis, sign);
				first = last;
			}
		}
}

struct ClumpBuild
{
	std::vector<Clump> clumps;
	int voxel_count = 0;
	std::vector<Quad> quads;
};

ClumpBuild build_clumps(Tree& tree, const Settings& settings, Skeleton& skeleton)
{
	ClumpBuild build;
	skeleton = Skeleton::from_tree(tree);
	build.clumps = place_clumps(skeleton, settings);
	if (build.clumps.empty())
		return build;

	double volume = 0.0;
	for (const Clump& clump : build.clumps)
		volume += solid_volume(clump, settings);
	double voxel = (double)settings.voxel;
	double estimate = volume / (voxel * voxel * voxel);
	double limit = ESTIMATE_POLYGON_MULTIPLE * (double)settings.max_polygons;
	if (estimate > limit)
		throw std::runtime_error("Leaf Clumps: an estimated " + std::to_string((long long)estimate) +
		                         " voxels exceeds 10 x Max Polygons (" +
		                         std::to_string(settings.max_polygons) + " polygons, " +
		                         std::to_string((long long)limit) + " voxels)");

	VoxelMap voxels = fill(build.clumps, settings);
	erode(voxels, settings);
	build.voxel_count = (int)voxels.size();

	std::vector<OwnedCell> cells = by_owner(voxels);
	for (size_t first = 0; first < cells.size();)
	{
		size_t last = first;
		while (last < cells.size() && cells[last].owner == cells[first].owner)
			last++;
		mesh_owner(build.quads, voxels, cells, first, last);
		first = last;
	}
	return build;
}

struct EmittedVertex
{
	int owner;
	Vector3 position;
};

struct Emission
{
	std::vector<EmittedVertex> vertices;
	std::vector<std::array<int, 4>> quads;
};

Emission emission_of(const ClumpBuild& build, float voxel)
{
	Emission emission;
	std::unordered_map<Cell, int, CellHash> vertex_of;
	int current_owner = -1;
	for (const Quad& quad : build.quads)
	{
		if (quad.owner != current_owner)
		{
			vertex_of.clear();
			current_owner = quad.owner;
		}
		std::array<int, 4> corners;
		for (size_t corner = 0; corner < 4; corner++)
		{
			const Cell& lattice = quad.corners[corner];
			auto found = vertex_of.find(lattice);
			if (found == vertex_of.end())
			{
				Vector3 position = Vector3((float)lattice[0], (float)lattice[1], (float)lattice[2]) * voxel;
				found = vertex_of.emplace(lattice, (int)emission.vertices.size()).first;
				emission.vertices.push_back(EmittedVertex{quad.owner, position});
			}
			corners[corner] = found->second;
		}
		emission.quads.push_back(corners);
	}
	return emission;
}

void emit(Mesh& mesh, const ClumpBuild& build, const Emission& emission, const Skeleton& skeleton)
{
	SegmentAttributes writer(mesh);
	int first = (int)mesh.vertices.size();
	for (const EmittedVertex& vertex : emission.vertices)
	{
		const Clump& clump = build.clumps[(size_t)vertex.owner];
		writer.add_vertex(vertex.position, skeleton.segments[(size_t)clump.segment], clump.radius,
		                  clump.attachment);
	}
	for (size_t quad = 0; quad < emission.quads.size(); quad++)
	{
		std::array<int, 4> corners = emission.quads[quad];
		for (int& corner : corners)
			corner += first;
		writer.add_quad(corners, build.quads[quad].uvs, LEAF_MATERIAL);
	}
}

int nearest_of(const std::vector<int>& candidates, const std::vector<EmittedVertex>& vertices,
               const Vector3& position)
{
	int best = -1;
	float best_distance = std::numeric_limits<float>::max();
	for (int candidate : candidates)
	{
		float distance = (vertices[(size_t)candidate].position - position).squaredNorm();
		if (distance < best_distance)
		{
			best_distance = distance;
			best = candidate;
		}
	}
	return best;
}

std::vector<int> lod0_sources(const Emission& lod, const Emission& lod0,
                              const std::vector<Clump>& clumps)
{
	std::vector<std::vector<int>> by_clump(clumps.size());
	std::unordered_map<int, std::vector<int>> by_segment;
	std::vector<int> all(lod0.vertices.size());
	for (int index = 0; index < (int)lod0.vertices.size(); index++)
	{
		int owner = lod0.vertices[(size_t)index].owner;
		by_clump[(size_t)owner].push_back(index);
		by_segment[clumps[(size_t)owner].segment].push_back(index);
		all[(size_t)index] = index;
	}

	std::vector<int> sources;
	sources.reserve(lod.vertices.size());
	for (const EmittedVertex& vertex : lod.vertices)
	{
		const std::vector<int>* candidates = &by_clump[(size_t)vertex.owner];
		if (candidates->empty())
		{
			auto segment = by_segment.find(clumps[(size_t)vertex.owner].segment);
			candidates = segment != by_segment.end() ? &segment->second : &all;
		}
		sources.push_back(nearest_of(*candidates, lod0.vertices, vertex.position));
	}
	return sources;
}
}

namespace Mtree
{

int LeafClumps::count_polygons(Tree& tree)
{
	clump_count = voxel_count = polygon_count = 0;
	Settings settings = sanitized(*this);
	Skeleton skeleton;
	ClumpBuild build = build_clumps(tree, settings, skeleton);
	clump_count = (int)build.clumps.size();
	voxel_count = build.voxel_count;
	polygon_count = (int)build.quads.size();
	return polygon_count;
}

void LeafClumps::append(Tree& tree, Mesh& mesh)
{
	clump_count = voxel_count = polygon_count = 0;
	Settings settings = sanitized(*this);
	Skeleton skeleton;
	ClumpBuild build = build_clumps(tree, settings, skeleton);
	clump_count = (int)build.clumps.size();
	voxel_count = build.voxel_count;
	polygon_count = (int)build.quads.size();
	if (polygon_count > settings.max_polygons)
		throw std::runtime_error("Leaf Clumps: " + std::to_string(polygon_count) +
		                         " polygons exceeds Max Polygons (" +
		                         std::to_string(settings.max_polygons) + ")");
	if (build.quads.empty())
		return;
	emit(mesh, build, emission_of(build, settings.voxel), skeleton);
}

std::vector<int> LeafClumps::append_lod(Tree& tree, Mesh& mesh, float voxel_scale)
{
	clump_count = voxel_count = polygon_count = 0;
	Settings lod0_settings = sanitized(*this);
	Skeleton skeleton;
	ClumpBuild lod0 = build_clumps(tree, lod0_settings, skeleton);

	Settings settings = lod0_settings;
	settings.voxel = std::clamp(lod0_settings.voxel * std::clamp(voxel_scale, 1.0f, 16.0f),
	                            lod0_settings.voxel, 16.0f);
	ClumpBuild build = build_clumps(tree, settings, skeleton);
	clump_count = (int)build.clumps.size();
	voxel_count = build.voxel_count;
	polygon_count = (int)build.quads.size();
	if (build.quads.empty())
		return {};

	Emission emission = emission_of(build, settings.voxel);
	std::vector<int> sources =
	    lod0_sources(emission, emission_of(lod0, lod0_settings.voxel), build.clumps);
	emit(mesh, build, emission, skeleton);
	return sources;
}

}
