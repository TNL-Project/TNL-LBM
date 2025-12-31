#pragma once

#include <magic_enum/magic_enum_utility.hpp>
#include <nanobind/nanobind.h>
#include <pytnl/pytnl.h>

template <typename MACRO, typename Scope>
void export_macro(Scope& scope, const char* name)
{
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
}
