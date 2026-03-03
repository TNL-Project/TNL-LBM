#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/format.h>
#include <TNL/Backend/Functions.h>

#include "lbm_common/fileutils.h"

enum class VelocityProfile : std::uint8_t
{
	zero,
	flat,
	sinus,
	block
};

void sinusVelocityProfile(double* velocityProfile, int sizeY, int sizeZ, int yBeg, int zBeg, int yEnd, int zEnd, double amplitude)
{
	int blockSizeY = yEnd - yBeg + 1;
	int blockSizeZ = zEnd - zBeg + 1;
	for (int k = zBeg; k <= zEnd; k++)
		for (int j = yBeg; j <= yEnd; j++) {
			if (j - yBeg < (int) (0.15 * (blockSizeY - 1)) + 1 && k - zBeg < (int) (0.15 * (blockSizeZ - 1)) + 1)
				velocityProfile[k * sizeY + j] = amplitude * std::pow(std::sin(M_PI * (double) (j - yBeg) / 0.3 / (double) (blockSizeY - 1)), 2)
											   * std::pow(std::sin(M_PI * (double) (k - zBeg) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else if (j - yBeg < (int) (0.15 * (blockSizeY - 1)) + 1 && k - zBeg > (int) (0.85 * (blockSizeZ - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * (double) (j - yBeg) / 0.3 / (double) (blockSizeY - 1)), 2)
					* std::pow(std::sin(M_PI * ((double) (k - zBeg) - (double) (blockSizeZ - 1)) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else if (j - yBeg > (int) (0.85 * (blockSizeY - 1)) && k - zBeg < (int) (0.15 * (blockSizeZ - 1)) + 1)
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (j - yBeg) - (double) (blockSizeY - 1)) / 0.3 / (double) (blockSizeY - 1)), 2)
					* std::pow(std::sin(M_PI * (double) (k - zBeg) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else if (j - yBeg > (int) (0.85 * (blockSizeY - 1)) && k - zBeg > (int) (0.85 * (blockSizeZ - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (j - yBeg) - (double) (blockSizeY - 1)) / 0.3 / (double) (blockSizeY - 1)), 2)
					* std::pow(std::sin(M_PI * ((double) (k - zBeg) - (double) (blockSizeZ - 1)) / 0.3 / (double) (blockSizeZ - 1)), 2);
			// y
			else if (j - yBeg < (int) (0.15 * (blockSizeY - 1)) + 1)
				velocityProfile[k * sizeY + j] = amplitude * std::pow(std::sin(M_PI * (double) (j - yBeg) / 0.3 / (double) (blockSizeY - 1)), 2);
			else if (j - yBeg < (int) (0.15 * (blockSizeY - 1)) + 1)
				velocityProfile[k * sizeY + j] = amplitude * std::pow(std::sin(M_PI * (double) (j - yBeg) / 0.3 / (double) (blockSizeY - 1)), 2);
			else if (j - yBeg > (int) (0.85 * (blockSizeY - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (j - yBeg) - (double) (blockSizeY - 1)) / 0.3 / (double) (blockSizeY - 1)), 2);
			else if (j - yBeg > (int) (0.85 * (blockSizeY - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (j - yBeg) - (double) (blockSizeY - 1)) / 0.3 / (double) (blockSizeY - 1)), 2);
			// z
			else if (k - zBeg < (int) (0.15 * (blockSizeZ - 1)) + 1)
				velocityProfile[k * sizeY + j] = amplitude * std::pow(std::sin(M_PI * (double) (k - zBeg) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else if (k - zBeg < (int) (0.15 * (blockSizeZ - 1)) + 1)
				velocityProfile[k * sizeY + j] = amplitude * std::pow(std::sin(M_PI * (double) (k - zBeg) / 0.3 / (double) (sizeZ - 1)), 2);
			else if (k - zBeg > (int) (0.85 * (blockSizeZ - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (k - zBeg) - (double) (blockSizeZ - 1)) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else if (k - zBeg > (int) (0.85 * (blockSizeZ - 1)))
				velocityProfile[k * sizeY + j] =
					amplitude * std::pow(std::sin(M_PI * ((double) (k - zBeg) - (double) (blockSizeZ - 1)) / 0.3 / (double) (blockSizeZ - 1)), 2);
			else
				velocityProfile[k * sizeY + j] = amplitude;
		}
}

std::unique_ptr<double[]> initGuess(VelocityProfile type, int sizeY, int sizeZ, double amplitude = 0)
{
	std::unique_ptr<double[]> velocityProfile{new double[sizeY * sizeZ]};
	switch (type) {
		case VelocityProfile::zero:
			for (int j = 0; j < sizeY; j++)
				for (int k = 0; k < sizeZ; k++)
					velocityProfile[k * sizeY + j] = 0;
			break;
		case VelocityProfile::flat:
			for (int j = 0; j < sizeY; j++)
				for (int k = 0; k < sizeZ; k++)
					velocityProfile[k * sizeY + j] = amplitude;
			break;
		case VelocityProfile::sinus:
			sinusVelocityProfile(velocityProfile.get(), sizeY, sizeZ, 1, 1, sizeY - 2, sizeZ - 2, amplitude);
			break;
		case VelocityProfile::block:
			sinusVelocityProfile(velocityProfile.get(), sizeY, sizeZ, 1, 1, (int) (sizeY * 0.5), (int) (sizeZ * 0.5), amplitude);
		default:
			break;
	}
	return velocityProfile;
}

void saveVelocityProfile(const std::string& dirname, const double* velocityProfile, int sizeY, int sizeZ, char axis)
{
	mkdir_p(dirname.c_str(), 0777);
	const std::string fname = fmt::format("{}/velocityProfile{}", dirname, axis);
	FILE* f = fopen(fname.c_str(), "wb");
	if (f == nullptr) {
		throw std::runtime_error("unable to access file velocityProfile");
	}
	if (fwrite(velocityProfile, sizeof(double), sizeY * sizeZ, f) != (std::size_t) (sizeY * sizeZ)) {
		throw std::runtime_error("failed to write data to velocityProfile");
	}
	fclose(f);
}

void saveVelocityProfile_txt(const std::string& dirname, const double* velocityProfile, int sizeY, int sizeZ, char axis)
{
	mkdir_p(dirname.c_str(), 0777);
	const std::string fname = fmt::format("{}/velocityProfile{}.txt", dirname, axis);
	std::ofstream velocityFile(fname.c_str());
	for (int j = 0; j < sizeY; j++) {
		for (int k = 0; k < sizeZ; k++) {
			velocityFile << fmt::format("{:+.5f}", velocityProfile[k * sizeY + j]) << "\t\t";
		}
		velocityFile << "\n";
	}
	velocityFile.close();
}

void loadVelocityProfile(const std::string& dirname, double* velocityProfile, int sizeY, int sizeZ, char axis)
{
	const std::string fname = fmt::format("{}/velocityProfile{}", dirname, axis);
	FILE* f = fopen(fname.c_str(), "rb");
	if (f == nullptr) {
		throw std::runtime_error("unable to access file velocityProfile");
	}
	if (fread(velocityProfile, sizeof(double), sizeY * sizeZ, f) != (std::size_t) (sizeY * sizeZ)) {
		throw std::runtime_error("failed to read data from velocityProfile");
	}
	fclose(f);
}

template <typename dreal, typename idx>
void
allocateCopyVelocityProfile(dreal** velocity, idx sizeY, idx sizeZ, idx offsetY, idx offsetZ, const double* velocityDes, int sizeYDes, int sizeZDes)
{
#ifdef USE_CUDA
	#if defined(__CUDACC__)
	TNL_BACKEND_SAFE_CALL(cudaMalloc((void**) velocity, sizeY * sizeZ * sizeof(dreal)));
	#elif defined(__HIP__)
	TNL_BACKEND_SAFE_CALL(hipMalloc((void**) velocity, sizeY * sizeZ * sizeof(dreal)));
	#endif
#else
	*velocity = new dreal[sizeY * sizeZ];
#endif

#ifdef USE_CUDA
	// convert velocityDes solution from double to dreal
	std::unique_ptr<dreal[]> velocityDesDreal{new dreal[sizeY * sizeZ]};
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			velocityDesDreal[k * sizeY + j] = velocityDes[(offsetZ + k) * sizeYDes + (offsetY + j)];
	TNL::Backend::memcpy(*velocity, velocityDesDreal.get(), sizeY * sizeZ * sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
#else
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			(*velocity)[k * sizeY + j] = velocityDes[(offsetZ + k) * sizeYDes + (offsetY + j)];
#endif
}

template <typename dreal, typename idx>
void allocateCopyGradientProfile(dreal** gradient, idx sizeY, idx sizeZ)
{
#ifdef USE_CUDA
	#if defined(__CUDACC__)
	TNL_BACKEND_SAFE_CALL(cudaMalloc((void**) gradient, sizeY * sizeZ * sizeof(dreal)));
	#elif defined(__HIP__)
	TNL_BACKEND_SAFE_CALL(hipMalloc((void**) gradient, sizeY * sizeZ * sizeof(dreal)));
	#endif
#else
	*gradient = new dreal[sizeY * sizeZ];
#endif

#ifdef USE_CUDA
	// convert velocityDes solution from double to dreal
	std::unique_ptr<dreal[]> gradientDreal{new dreal[sizeY * sizeZ]};
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			gradientDreal[k * sizeY + j] = (dreal) 0;
	TNL::Backend::memcpy(*gradient, gradientDreal.get(), sizeY * sizeZ * sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
#else
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			(*gradient)[k * sizeY + j] = (dreal) 0;
#endif
}

template <typename dreal, typename idx>
void copyVelocityProfile(dreal* velocity, idx sizeY, idx sizeZ, idx offsetY, idx offsetZ, double* velocityDes, int sizeYDes, int sizeZDes)
{
#ifdef USE_CUDA
	// convert velocityDes solution from dreal to double
	std::unique_ptr<dreal[]> velocityDouble{new dreal[sizeY * sizeZ]};
	TNL::Backend::memcpy(velocityDouble.get(), velocity, sizeY * sizeZ * sizeof(dreal), TNL::Backend::MemcpyDeviceToHost);
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			velocityDes[(offsetZ + k) * sizeYDes + (offsetY + j)] = (double) velocityDouble[k * sizeY + j];
#else
	for (idx j = 0; j < sizeY; j++)
		for (idx k = 0; k < sizeZ; k++)
			velocityDes[(offsetZ + k) * sizeYDes + (offsetY + j)] = (double) velocity[k * sizeY + j];
#endif
}

template <typename dreal>
void deallocateVelocityProfile(dreal** velocity)
{
	if (*velocity) {
#ifdef USE_CUDA
	#if defined(__CUDACC__)
		TNL_BACKEND_SAFE_CALL(cudaFree(*velocity));
	#elif defined(__HIP__)
		TNL_BACKEND_SAFE_CALL(hipFree(*velocity));
	#endif
#else
		delete[] (*velocity);
#endif
	}
}

template <typename dreal>
void deallocateGradientProfile(dreal** gradient)
{
	if (*gradient) {
#ifdef USE_CUDA
	#if defined(__CUDACC__)
		TNL_BACKEND_SAFE_CALL(cudaFree(*gradient));
	#elif defined(__HIP__)
		TNL_BACKEND_SAFE_CALL(hipFree(*gradient));
	#endif
#else
		delete[] (*gradient);
#endif
	}
}
