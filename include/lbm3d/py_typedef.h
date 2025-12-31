#pragma once

#include <pytnl/pytnl.h>

template <typename T, typename... C>
void export_typedef(nb::class_<C...>& m, const char* name)
{
	m.def_prop_ro_static(
		name,
		[](nb::handle) -> nb::typed<nb::handle, nb::type_object>
		{
			// nb::type<> does not handle generic types like int, float, etc.
			// https://github.com/wjakob/nanobind/discussions/1070
			if constexpr (std::is_same_v<T, bool>) {
				return nb::borrow(&PyBool_Type);
			}
			else if constexpr (std::is_integral_v<T>) {
				return nb::borrow(&PyLong_Type);
			}
			else if constexpr (std::is_floating_point_v<T>) {
				return nb::borrow(&PyFloat_Type);
			}
			else if constexpr (TNL::is_complex_v<T>) {
				return nb::borrow(&PyComplex_Type);
			}
			else {
				return nb::type<T>();
			}
		}
	);
}
