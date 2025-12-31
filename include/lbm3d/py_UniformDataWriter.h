#pragma once

#include <pytnl/pytnl.h>

#include "UniformDataWriter.h"

template <typename TRAITS>
void export_UniformDataWriter(nb::module_& m, const char* name)
{
	using UniformDataWriter = ::UniformDataWriter<TRAITS>;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using point_t = typename TRAITS::point_t;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using xyz_ndarray = nb::ndarray<const dreal, nb::ndim<3>, nb::device::cpu>;

	auto writer =  //
		nb::class_<UniformDataWriter>(m, name)
			//.def(nb::init<idx3d, idx3d, idx3d, point_t, real, DataManager, std::string>())
			.def(
				"write",
				[](UniformDataWriter& self, const std::string& varName, real value)
				{
					self.write(varName, value);
				},
				nb::arg("varName"),
				nb::arg("value")
			)
			.def(
				"write",
				[](UniformDataWriter& self, const std::string& varName, idx value)
				{
					self.write(varName, value);
				},
				nb::arg("varName"),
				nb::arg("value")
			)
			.def(
				"write",
				[](UniformDataWriter& self, const std::string& varName, xyz_ndarray src, idx3d begin, idx3d end)
				{
					// check if begin matches the size of the src array
					if (begin.x() < 0 || begin.y() < 0 || begin.z() < 0 || begin.x() >= idx(src.shape(0)) || begin.y() >= idx(src.shape(1))
						|| begin.z() >= idx(src.shape(2)))
						throw std::invalid_argument(
							fmt::format(
								"Begin indices ({}, {}, {}) do not match the size of the src array ({}, {}, {}).",
								begin.x(),
								begin.y(),
								begin.z(),
								src.shape(0),
								src.shape(1),
								src.shape(2)
							)
						);
					// check if end matches the size of the src array
					if (end.x() < 0 || end.y() < 0 || end.z() < 0 || end.x() > idx(src.shape(0)) || end.y() > idx(src.shape(1))
						|| end.z() > idx(src.shape(2)))
						throw std::invalid_argument(
							fmt::format(
								"end indices ({}, {}, {}) do not match the size of the src array ({}, {}, {}).",
								end.x(),
								end.y(),
								end.z(),
								src.shape(0),
								src.shape(1),
								src.shape(2)
							)
						);

					self.write(
						varName,
						[&](idx x, idx y, idx z) -> dreal
						{
							return src(x, y, z);
						},
						begin,
						end
					);
				},
				nb::arg("varName"),
				nb::arg("src"),
				nb::arg("begin"),
				nb::arg("end")
			)
		//
		;
}
