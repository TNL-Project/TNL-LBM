#pragma once

#include <magic_enum/magic_enum_utility.hpp>
#include <nanobind/nanobind.h>
#include <pytnl/pytnl.h>

#include "py_macro.h"  // py_aux_type_cache

template <typename BC, typename Scope>
void export_bc(Scope& scope, const char* name)
{
	// same type-indexed cache as export_macro: a BC type shared by multiple
	// configs is registered once and attached to the later scopes
	auto& cache = py_aux_type_cache();
	const std::type_index key(typeid(BC));
	if (const auto it = cache.find(key); it != cache.end()) {
		scope.attr(name) = it->second;
		return;
	}

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
	cache.emplace(key, nb::borrow<nb::object>(macro.ptr()));
}
