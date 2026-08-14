/*
 * doctest implementation translation unit for the unit-test binary.
 *
 * Defines the doctest runner symbols (DOCTEST_CONFIG_IMPLEMENT) and the
 * custom main(): MPI must be initialized before doctest runs (the test
 * cases construct LBM_BLOCK instances), so doctest's own main cannot be
 * used. Every other .cu file in this directory is a plain doctest
 * test-case TU linked into the same test_cpp_units binary.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "lbm3d/defs.h"

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);
	return doctest::Context(argc, argv).run();
}
