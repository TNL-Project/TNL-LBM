#pragma once

#include <adios2.h>
#include <fmt/core.h>

class CheckpointManager
{
private:
	adios2::IO io;
	adios2::Engine engine;
	adios2::Mode mode = adios2::Mode::Undefined;

public:
	CheckpointManager(adios2::ADIOS& adios)
	{
		io = adios.DeclareIO("io");
		io.SetEngine("File");
	}

	// Open an engine and begin a logical ADIOS2 step for the checkpoint.
	// Note that this is an MPI collective operation.
	void start(const std::string filename, adios2::Mode mode)
	{
		engine = io.Open(filename, mode);
		engine.BeginStep();
		this->mode = mode;
	}

	// Perform all deferred put/get operations (depending on the current mode).
	void performDeferred()
	{
		if (mode == adios2::Mode::Read)
			engine.PerformGets();
		else
			engine.PerformPuts();
	}

	// End the current ADIOS2 step and close the engine.
	// Note that this is an MPI collective operation.
	void finalize()
	{
		engine.EndStep();
		engine.Close();
		mode = adios2::Mode::Undefined;
	}

	template <typename T, typename CastToType = T>
	void saveLoadAttribute(const std::string& name, T& variable)
	{
		if (mode == adios2::Mode::Write)
			// We are not re-declaring the IO object, so we must specify `true`
			// for the `allowModification` parameter in `DefineAttribute`.
			io.DefineAttribute<CastToType>(name, static_cast<CastToType>(variable), "", "/", true);
		else
			variable = static_cast<T>(io.InquireAttribute<CastToType>(name).Data()[0]);
	}

	template <typename LBM_BLOCK, typename Array>
	void saveLoadVariable(std::string name, LBM_BLOCK& block, Array& array)
	{
		using T = typename Array::ValueType;

		// NOTE: For checkpointing functionality we need to save overlaps
		// which don't map nicely to the ADIOS2 model. Hence, we save/load
		// each block as a separate variable.
		adios2::Dims shape;
		adios2::Dims start;
		adios2::Dims count;
		//if constexpr (Array::getDimension() == 4) {
		//	const std::size_t N = array.template getSize<0>();
		//	shape = adios2::Dims({N, size_t(block.global.z()), size_t(block.global.y()), size_t(block.global.x())});
		//	start = adios2::Dims({0, size_t(block.offset.z()), size_t(block.offset.y()), size_t(block.offset.x())});
		//	count = adios2::Dims({N, size_t(block.local.z()), size_t(block.local.y()), size_t(block.local.x())});
		//}
		//else {
		//	shape = adios2::Dims({size_t(block.global.z()), size_t(block.global.y()), size_t(block.global.x())});
		//	start = adios2::Dims({size_t(block.offset.z()), size_t(block.offset.y()), size_t(block.offset.x())});
		//	count = adios2::Dims({size_t(block.local.z()), size_t(block.local.y()), size_t(block.local.x())});
		//}
		start = {0};
#ifdef HAVE_MPI
		shape = count = {std::size_t(array.getLocalStorageSize())};
#else
		shape = count = {std::size_t(array.getStorageSize())};
#endif
		name += fmt::format("_block_{}", block.id);

		if (mode == adios2::Mode::Write) {
			// ADIOS2 variables cannot be redefined and we are not re-declaring the IO
			// object, so we must use InquireVariable.
			adios2::Variable<T> var = io.InquireVariable<T>(name);
			if (! var)
				var = io.DefineVariable<T>(name, shape, start, count);
			engine.Put(var, array.getData());
		}
		else {
			adios2::Variable<T> var = io.InquireVariable<T>(name);
			var.SetSelection({start, count});
			engine.Get(var, array.getData());
		}
	}

	// This is for the case where each rank maintains its own, independent array
	// (i.e., not DistributedNDArray).
	template <typename Array>
	void saveLoadLocalArray(std::string name, int rank, Array& array)
	{
		using T = typename Array::ValueType;

		// Note: NDArray dimensions could be mapped to adios2::Dims, but working with 1D works too...
		adios2::Dims shape = {std::size_t(array.getStorageSize())};
		adios2::Dims start = {0};
		adios2::Dims count = shape;
		name += fmt::format("_rank_{}", rank);

		if (mode == adios2::Mode::Write) {
			// ADIOS2 variables cannot be redefined and we are not re-declaring the IO
			// object, so we must use InquireVariable.
			adios2::Variable<T> var = io.InquireVariable<T>(name);
			if (! var)
				var = io.DefineVariable<T>(name, shape, start, count);
			engine.Put(var, array.getData());
		}
		else {
			adios2::Variable<T> var = io.InquireVariable<T>(name);
			var.SetSelection({start, count});
			engine.Get(var, array.getData());
		}
	}
};
