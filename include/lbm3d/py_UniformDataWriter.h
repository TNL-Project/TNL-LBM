#pragma once

#include <pytnl/pytnl.h>

#include "defs.h"
#include "UniformDataWriter.h"

template <typename TRAITS, typename xyz_ndarray>
void _uniformDataWriterWrite(
	::UniformDataWriter<TRAITS>& self,
	const std::string& varName,
	const xyz_ndarray& src,
	typename TRAITS::idx3d src_shape,
	typename TRAITS::idx3d begin,
	typename TRAITS::idx3d end
)
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	// check if begin matches the size of the src array
	if (begin.x() < 0 || begin.y() < 0 || begin.z() < 0 || begin.x() >= src_shape.x() || begin.y() >= src_shape.y() || begin.z() >= src_shape.z())
		throw std::invalid_argument(
			fmt::format(
				"Begin indices ({}, {}, {}) do not match the size of the src array ({}, {}, {}).",
				begin.x(),
				begin.y(),
				begin.z(),
				src_shape.x(),
				src_shape.y(),
				src_shape.z()
			)
		);
	// check if end matches the size of the src array
	if (end.x() < 0 || end.y() < 0 || end.z() < 0 || end.x() > src_shape.x() || end.y() > src_shape.y() || end.z() > src_shape.z())
		throw std::invalid_argument(
			fmt::format(
				"end indices ({}, {}, {}) do not match the size of the src array ({}, {}, {}).",
				end.x(),
				end.y(),
				end.z(),
				src_shape.x(),
				src_shape.y(),
				src_shape.z()
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
}

template <typename TRAITS>
void export_UniformDataWriter(nb::module_& m, const char* name)
{
	using UniformDataWriter = ::UniformDataWriter<TRAITS>;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using xyz_ndarray = nb::ndarray<const dreal, nb::ndim<3>, nb::device::cpu>;
	// special subarray view used in outputData
	using special_macro_view_t = decltype(getMacroView<TRAITS, typename TRAITS::hmacro_array_t>(typename TRAITS::hmacro_array_t(), 0));

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
				[](UniformDataWriter& self, const std::string& varName, const xyz_ndarray& src, idx3d begin, idx3d end)
				{
					idx3d src_shape{idx(src.shape(0)), idx(src.shape(1)), idx(src.shape(2))};
					_uniformDataWriterWrite(self, varName, src, src_shape, begin, end);
				},
				nb::arg("varName"),
				nb::arg("src"),
				nb::arg("begin"),
				nb::arg("end")
			)
			.def(
				"write",
				[](UniformDataWriter& self, const std::string& varName, const special_macro_view_t& src, idx3d begin, idx3d end)
				{
					idx3d src_shape{src.getSizes()[0], src.getSizes()[1], src.getSizes()[2]};
					_uniformDataWriterWrite(self, varName, src, src_shape, begin, end);
				},
				nb::arg("varName"),
				nb::arg("src"),
				nb::arg("begin"),
				nb::arg("end")
			)
		//
		;
}
