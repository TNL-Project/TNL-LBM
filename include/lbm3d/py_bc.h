#pragma once

#include <magic_enum/magic_enum_utility.hpp>
#include <nanobind/nanobind.h>
#include <pytnl/pytnl.h>

template <typename BC, typename Scope>
void export_bc(Scope& scope, const char* name)
{
	auto macro = nb::class_<BC>(scope, name);
	auto GEO = nb::enum_<typename BC::GEO>(macro, "GEO", nb::is_arithmetic());
	magic_enum::enum_for_each<typename BC::GEO>(
		[&GEO](auto val) mutable
		{
			constexpr typename BC::GEO geo = val;
			const std::string_view& name_view = magic_enum::enum_name(geo);
			const std::string name(name_view.data(), name_view.size());
			GEO.value(name.c_str(), geo);
		}
	);
	GEO.export_values();
}
