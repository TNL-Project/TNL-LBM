#pragma once

#include <stdexcept>

#include <adios2.h>
#include <fmt/core.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/DistributedNDArrayView.h>

#include "lbm3d/checkpoint.h"

// saves or loads macro values
template <typename State>
void saveloadMacro(State& state, adios2::Mode mode, const std::string& fname, bool steady = false)
{
	using MACRO = typename State::MACRO;
	static_assert(MACRO::N > 0);

	adios2::IO io;
	try {
		io = state.adios.AtIO("macro");
	}
	catch (std::invalid_argument& e) {
		io = state.adios.DeclareIO("macro");
		io.SetEngine("bpfile");
	}

	adios2::Engine engine = io.Open(fname, mode);
	engine.BeginStep();

	for (auto& block : state.nse.blocks) {
		checkpoint_global_array(io, engine, mode, "LBM_macro", block, block.hmacro);
	}

	engine.EndStep();
	engine.Close();
}

// loads primary and measured macro values - adjoint problem
template <typename State>
void loadPrimaryAndMeasuredMacro(State& state, const std::string& fname_primary, const std::string& fname_measured, bool steady = false)
{
	if (!steady) {
		throw std::runtime_error("loadPrimaryAndMeasuredMacro is not implemented yet for 'steady = false'");
	}

	using MACRO = typename State::MACRO;
	static_assert(MACRO::N > 0);
	using dreal = typename State::TRAITS::dreal;
	using local_array4d_view = typename State::TRAITS::template array4d<MACRO::N / 2, dreal, TNL::Devices::Host>::ViewType;
#ifdef HAVE_MPI
	using array4d_view = TNL::Containers::DistributedNDArrayView<local_array4d_view>;
#else
	using array4d_view = local_array4d_view;
#endif
	using SizesHolder = typename local_array4d_view::SizesHolderType;
	using hmacro_array_t = typename State::BLOCK_NSE::hmacro_array_t;

	adios2::IO io;
	try {
		io = state.adios.AtIO("macro");
	}
	catch (std::invalid_argument& e) {
		io = state.adios.DeclareIO("macro");
		io.SetEngine("bpfile");
	}
	adios2::Mode mode = adios2::Mode::Read;

	adios2::Engine engine = io.Open(fname_primary, mode);
	engine.BeginStep();

	for (auto& block : state.nse.blocks) {
		// primary macros - first half of block.hmacro
#ifdef HAVE_MPI
		auto hmacro = block.hmacro.getLocalView();
#else
		auto hmacro = block.hmacro.getView();
#endif
		SizesHolder sizes;	// first dimension is static: N/2
		sizes.template setSize<1>(hmacro.template getSize<1>());
		sizes.template setSize<2>(hmacro.template getSize<2>());
		sizes.template setSize<3>(hmacro.template getSize<3>());
		auto strides = hmacro.getStrides();
		auto overlaps = hmacro.getOverlaps();
		typename local_array4d_view::IndexerType indexer(sizes, strides, overlaps);

		array4d_view macro_view;
		dreal* begin = block.hmacro.getData();
		macro_view.bind(begin, indexer);

		checkpoint_global_array(io, engine, mode, "LBM_macro", block, macro_view);
	}

	engine.EndStep();
	engine.Close();

	engine = io.Open(fname_measured, mode);
	engine.BeginStep();

	for (auto& block : state.nse.blocks) {
		// measured macros - second half of block.hmacro
#ifdef HAVE_MPI
		auto hmacro = block.hmacro.getLocalView();
#else
		auto hmacro = block.hmacro.getView();
#endif
		SizesHolder sizes;	// first dimension is static: N/2
		sizes.template setSize<1>(hmacro.template getSize<1>());
		sizes.template setSize<2>(hmacro.template getSize<2>());
		sizes.template setSize<3>(hmacro.template getSize<3>());
		auto strides = hmacro.getStrides();
		auto overlaps = hmacro.getOverlaps();
		typename local_array4d_view::IndexerType indexer(sizes, strides, overlaps);

		array4d_view macro_view;
		dreal* begin = block.hmacro.getData() + hmacro.getStorageSize() / 2;
		macro_view.bind(begin, indexer);

		checkpoint_global_array(io, engine, mode, "LBM_macro", block, macro_view);
	}

	engine.EndStep();
	engine.Close();
}
