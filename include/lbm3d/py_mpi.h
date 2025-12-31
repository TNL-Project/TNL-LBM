#pragma once

// Based on https://github.com/mpi4py/mpi4py/blob/b76b46d01c78588e8843f2279473c71cd3b881f6/demo/wrap-nanobind/helloworld.cxx

#ifdef HAVE_MPI
	#include <mpi.h>
	#define MPI4PY_LIMITED_API 1
	#define MPI4PY_LIMITED_API_SKIP_MESSAGE 1
	#define MPI4PY_LIMITED_API_SKIP_SESSION 1
	#include <mpi4py/mpi4py.h>
	#include <nanobind/nanobind.h>
#else
	#include <TNL/MPI/DummyDefs.h>
#endif

namespace nb = nanobind;

inline MPI_Comm py2mpi(nb::object obj)
{
#ifdef HAVE_MPI
	// Lazy-import mpi4py to define the PyMPIComm_Get function
	if (PyMPIComm_Get == nullptr) {
		if (import_mpi4py() < 0)
			throw nb::python_error();
	}

	PyObject* pyobj = obj.ptr();
	MPI_Comm* mpi_ptr = PyMPIComm_Get(pyobj);
	if (mpi_ptr == nullptr)
		throw nb::python_error();
	return *mpi_ptr;
#else
	return MPI_COMM_WORLD;
#endif
}
