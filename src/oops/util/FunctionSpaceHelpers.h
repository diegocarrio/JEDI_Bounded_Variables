/*
 * (C) Copyright 2024 UCAR
 * (C) Crown Copyright 2024 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <utility>
#include <vector>

#include "atlas/functionspace.h"

#include "oops/util/Logger.h"

namespace atlas {
namespace grid { class Distribution; }
namespace grid { class Partitioner; }
class Grid;
class Mesh;
class FunctionSpace;
class FieldSet;
}  // namespace atlas

namespace eckit {
namespace mpi { class Comm; }
class Configuration;
}  // namespace eckit

namespace util {

/// \brief procedure to call a functor for a given concrete implementation
///        of a function space type
template<typename Functor>
void executeFunc(const atlas::FunctionSpace & fspace, const Functor & functor) {
  if (atlas::functionspace::NodeColumns(fspace)) {
    functor(atlas::functionspace::CubedSphereNodeColumns(fspace));
  } else if (atlas::functionspace::StructuredColumns(fspace)) {
    functor(atlas::functionspace::StructuredColumns(fspace));
  } else {
    oops::Log::error() << "ERROR - a functor call failed "
                          "(function space type not allowed)" << std::endl;
    throw std::runtime_error("a functor call failed");
  }
}

atlas::idx_t getSizeOwned(const atlas::FunctionSpace & fspace);

// -----------------------------------------------------------------------------

// This class computes and caches the mapping between atlas's two distinct indexings into the Mesh
// and the StructuredColumns FunctionSpace that can be built from the same StructuredGrid. The Mesh
// and the FunctionSpace will agree on the indexing of the owned grid points, but will in general
// disagree on the number AND indexing of the halo grid points.
//
// This class converts a mesh index into a FunctionSpace (FieldSet) index when possible, i.e., when
// both the Mesh and the FunctionSpace agree a particular point is part of the halo. If a Mesh ghost
// point is not part of the FunctionSpace halo, then the class returns a missing value. If a
// FunctionSpace ghost point is not part of the Mesh halo, then the class is not even aware of this
// point.
//
// WARNING: for a global StructuredColumns FunctionSpace, the Mesh generated by the
// StructuredMeshGenerator will likely be missing points at the lon=0/lon=360 meridian, so this is
// a location where this class will fail to map the Mesh halo to the FunctionSpace halo.
// See: https://github.com/ecmwf/atlas/issues/200
class StructuredMeshToStructuredColumnsIndexMap {
 public:
  // Default constructor produces an empty class
  StructuredMeshToStructuredColumnsIndexMap() = default;

  // Compute the mapping from Mesh indices to StructuredColumns indices
  void initialize(const atlas::Mesh &, const atlas::functionspace::StructuredColumns &);

  // Given an index into the Mesh, return the index into the StructuredColumns
  atlas::idx_t operator()(const atlas::idx_t mesh_index) const;

 private:
  bool valid_ = false;
  atlas::idx_t nb_mesh_owned_ = 0;
  atlas::idx_t nb_mesh_ghost_ = 0;
  std::vector<atlas::idx_t> map_ = {};
};

// -----------------------------------------------------------------------------

// Parses config and sets up an atlas::FunctionSpace and associated geometric data
void setupFunctionSpace(const eckit::mpi::Comm & comm,
    const eckit::Configuration & config,
    atlas::Grid & grid,
    atlas::grid::Partitioner & partitioner,
    atlas::Mesh & mesh,
    atlas::FunctionSpace & functionspace,
    atlas::FieldSet & fieldset);

// -----------------------------------------------------------------------------

// Define a distribution and a mesh from a custom partition
void setupStructuredMeshWithCustomPartition(const eckit::mpi::Comm &,
    const atlas::Grid &,
    const std::vector<int> &,
    atlas::grid::Distribution &,
    atlas::Mesh &);

// -----------------------------------------------------------------------------

}  // namespace util
