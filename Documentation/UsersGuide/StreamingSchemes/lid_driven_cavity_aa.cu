#include "../common/lbm_doc_common.h"

using Collision = D3Q27_SRT<TraitsSP>;

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("lid_driven_cavity_aa");
	tnl_lbm::doc::setupLidDrivenCavityCLI(program);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		fmt::println(stderr, "{}", err.what());
		std::cerr << program << '\n';
		return 1;
	}

	auto options = tnl_lbm::doc::parseLidDrivenCavityOptions(program);

// [doc-lid-aa-run-start]
    return tnl_lbm::doc::runLidDrivenCavity<TraitsSP, Collision>("lid_driven_cavity_aa", std::move(options));
// [doc-lid-aa-run-end]
}
