#pragma once

#include <spdlog/common.h>
#include <sys/stat.h>  // mkdir(2)
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <TNL/MPI/Comm.h>

static inline spdlog::sink_ptr init_file_sink(const std::string& name, const std::string& id, const TNL::MPI::Comm& communicator)
{
	const int rank = TNL::MPI::GetRank(communicator);

	// create the output directory for logs
	const std::string dir = fmt::format("results_{}", id);
	mkdir(dir.c_str(), 0777);
	const std::string log_fname = fmt::format("{}/log_{}_rank{:03d}", dir, name, rank);

	// initialize the file sink
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_fname);
	file_sink->set_level(spdlog::level::trace);
	file_sink->set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S %z] [rank {:d}, thread %t] [%l] %v", rank));

	return file_sink;
}

static inline void init_file_logger(const std::string& name, const std::string& id, const TNL::MPI::Comm& communicator)
{
	auto file_sink = init_file_sink(name, id, communicator);
	auto logger = std::make_shared<spdlog::logger>(name, file_sink);
	logger->set_level(spdlog::level::trace);

	// globally register the loggers so they can be accessed using spdlog::get(name)
	spdlog::register_logger(logger);
}

static inline void init_logging(const std::string& id, const TNL::MPI::Comm& communicator)
{
	// initialize the file sink
	auto file_sink = init_file_sink("main", id, communicator);

	// initialize the console sink
	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	console_sink->set_level(spdlog::level::info);
	console_sink->set_pattern("[%^%l%$] %v");

	// initialize the logger
	spdlog::sinks_init_list sink_list = {console_sink, file_sink};
	auto logger = std::make_shared<spdlog::logger>("main", sink_list);
	logger->set_level(spdlog::level::trace);
	logger->flush_on(spdlog::level::info);

	// globally register the loggers so they can be accessed using spdlog::get(name)
	spdlog::register_logger(logger);

	// set the logger as global default logger
	spdlog::set_default_logger(logger);

	// initialize spdlog logger for profiling output
	init_file_logger("profile", id, communicator);
}

static inline void deinit_logging()
{
	// create new default logger (same as spdlog itself) before destroying the current one
	auto color_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	const char* default_logger_name = "";
	auto logger = std::make_shared<spdlog::logger>(default_logger_name, std::move(color_sink));
	spdlog::set_default_logger(std::move(logger));

	// drop/unregister all loggers initialized by our code
	spdlog::drop("main");
	spdlog::drop("profile");
}
