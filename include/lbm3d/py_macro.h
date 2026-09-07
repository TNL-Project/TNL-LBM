#pragma once

#include <magic_enum/magic_enum_utility.hpp>
#include <nanobind/nanobind.h>
#include <pytnl/pytnl.h>

#include <typeindex>
#include <unordered_map>

// Module-level cache of already-registered auxiliary types: BC/MACRO are
// TRAITS- or config-family-level types shared by every instantiation that
// reuses them, but nanobind registers each C++ type at most once - a later
// export simply attaches the cached class under the new scope's name
inline std::unordered_map<std::type_index, nb::object>& py_aux_type_cache()
{
	static std::unordered_map<std::type_index, nb::object> cache;
	return cache;
}

template <typename MACRO, typename Scope>
void export_macro(Scope& scope, const char* name)
{
	auto& cache = py_aux_type_cache();
	const std::type_index key(typeid(MACRO));
	if (const auto it = cache.find(key); it != cache.end()) {
		scope.attr(name) = it->second;
		return;
	}

	auto macro = nb::class_<MACRO>(scope, name);
	auto QuantityNames = nb::enum_<typename MACRO::QuantityNames>(macro, "QuantityNames", nb::is_arithmetic());
	magic_enum::enum_for_each<typename MACRO::QuantityNames>(
		[&QuantityNames](auto val) mutable
		{
			constexpr typename MACRO::QuantityNames q_name = val;
			const std::string_view& name_view = magic_enum::enum_name(q_name);
			const std::string name(name_view.data(), name_view.size());
			QuantityNames.value(name.c_str(), q_name);
		}
	);
	QuantityNames.export_values();
	cache.emplace(key, nb::borrow<nb::object>(macro.ptr()));
}
