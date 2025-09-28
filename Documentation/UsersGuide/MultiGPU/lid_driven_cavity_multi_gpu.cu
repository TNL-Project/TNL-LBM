#include "../common/lbm_doc_common.h"

#ifdef __CUDACC__
	#include <cuda_runtime.h>
#endif

using Collision = D3Q27_SRT<TraitsSP>;

namespace
{
// [doc-mgpu-select-start]
void selectDeviceForRank(int requestedDevice)
{
#ifdef __CUDACC__
	if (requestedDevice >= 0) {
		cudaError_t status = cudaSetDevice(requestedDevice);
		if (status != cudaSuccess)
			throw std::runtime_error(fmt::format("cudaSetDevice({}) failed: {}", requestedDevice, cudaGetErrorString(status)));
		spdlog::info("Using user-selected GPU {}", requestedDevice);
		return;
	}

	int deviceCount = 0;
	cudaError_t status = cudaGetDeviceCount(&deviceCount);
	if (status != cudaSuccess || deviceCount == 0) {
		spdlog::warn("No CUDA devices detected, relying on default device selection.");
		return;
	}

	const int rank = TNL::MPI::GetRank(MPI_COMM_WORLD);
	const int device = rank % deviceCount;
	status = cudaSetDevice(device);
	if (status != cudaSuccess)
		throw std::runtime_error(fmt::format("cudaSetDevice({}) failed: {}", device, cudaGetErrorString(status)));

	spdlog::info("Rank {} mapped to GPU {} of {} available devices", rank, device, deviceCount);
#else
	(void) requestedDevice;
#endif
}
// [doc-mgpu-select-end]
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("lid_driven_cavity_multi_gpu");
	tnl_lbm::doc::setupLidDrivenCavityCLI(program);
	program.add_argument("--local-device")
		.help("force a specific GPU id on this rank")
		.scan<'i', int>()
		.default_value(-1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		fmt::println(stderr, "{}", err.what());
		std::cerr << program << '\n';
		return 1;
	}

	const int requestedDevice = program.get<int>("--local-device");

	selectDeviceForRank(requestedDevice);

	auto options = tnl_lbm::doc::parseLidDrivenCavityOptions(program);

// [doc-mgpu-run-start]
	return tnl_lbm::doc::runLidDrivenCavity<TraitsSP, Collision>("lid_driven_cavity_multi_gpu", std::move(options));
// [doc-mgpu-run-end]
}
