/*
 * doctest runner for the single-rank AMR test binaries (unit + driver).
 *
 * Same MPI-before-doctest setup as doctest_main.cu, plus the single-rank
 * fail-fast guard of the retired per-suite mains: on multiple ranks each
 * rank would execute every census State and double-write the writer's
 * vtkhdf outputs, degrading the caller's one-answer contract silently.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <fmt/core.h>

#include "lbm3d/defs.h"

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 1) {
		fmt::println("RESULT: AMR doctest suites are single-rank only (nproc = {})", TNL::MPI::GetSize(MPI_COMM_WORLD));
		return 1;
	}

	return doctest::Context(argc, argv).run();
}
