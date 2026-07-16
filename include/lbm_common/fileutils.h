#pragma once

#include <unistd.h>	 // access

static bool fileExists(const char* fname)
{
	//FILE *fp = fopen(fname, "r");
	//if (!fp) return false;
	//fclose(fp);
	//return true;

	// POSIX way is faster
	return access(fname, F_OK) == 0;
}

#include <string.h>
#include <linux/limits.h>  // PATH_MAX
#include <sys/stat.h>	   // mkdir(2)
#include <errno.h>

// adapted from http://stackoverflow.com/a/2336245/119527
static int mkdir_p(const char* path, mode_t mode)
{
	const size_t len = strlen(path);
	char _path[PATH_MAX];
	char* p;

	errno = 0;

	// copy string so it is mutable
	if (len > sizeof(_path) - 1) {
		errno = ENAMETOOLONG;
		return -1;
	}
	strcpy(_path, path);

	// iterate the string
	for (p = _path + 1; *p; p++) {
		if (*p == '/') {
			// temporarily truncate
			*p = '\0';

			if (mkdir(_path, mode) != 0) {
				if (errno != EEXIST)
					return -1;
			}

			*p = '/';
		}
	}

	if (mkdir(_path, mode) != 0) {
		if (errno != EEXIST)
			return -1;
	}

	return 0;
}

#include <libgen.h>	 // dirname, basename

// create parent directories of a file path
static int create_parent_directories(const char* fname)
{
	char buffer[PATH_MAX];
	strcpy(buffer, fname);
	char* dir = dirname(buffer);
	return mkdir_p(dir, 0777);
}

#include <stdio.h>	// FILE, fopen, fclose

// create parent directories and then the file
static int create_file(const char* fname)
{
	// return early if the file already exists
	if (fileExists(fname))
		return 0;

	// make sure that the parent directory exists
	create_parent_directories(fname);

	// create the file
	FILE* fp = fopen(fname, "wb");
	if (fp == NULL) {
		fprintf(stderr, "error: failed to create file %s: %s\n", fname, strerror(errno));
		return -1;
	}
	fclose(fp);

	return 0;
}

#include <stdio.h>	 // renameat2
#include <fcntl.h>	 // open
#include <unistd.h>	 // close
#include <error.h>	 // errno

// swap two filenames on the same filesystem https://lwn.net/Articles/569134/
static int rename_exchange(const char* oldpath, const char* newpath)
{
	// renameat2 is available since glibc 2.28
	// We need to emulate a workaround for Helios ;-(
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 28))
	int result = renameat2(AT_FDCWD, oldpath, AT_FDCWD, newpath, RENAME_EXCHANGE);
	if (errno == ENOENT) {
		// newpath does not exist, cannot use RENAME_EXCHANGE
		result = renameat2(AT_FDCWD, oldpath, AT_FDCWD, newpath, RENAME_NOREPLACE);
	}
	return result;
#else
	// if the target does not exist, just move it
	if (! fileExists(newpath))
		return renameat(AT_FDCWD, oldpath, AT_FDCWD, newpath);
	// make a temporary path
	char buffer[PATH_MAX];
	strcpy(buffer, newpath);
	strcat(buffer, "_tmp_for_exchange");
	if (fileExists(buffer)) {
		spdlog::error("temporary path {} already exists, cannot exchange files", buffer);
		return -1;
	}
	// move newpath to the buffer
	int result = renameat(AT_FDCWD, newpath, AT_FDCWD, buffer);
	if (result != 0)
		return result;
	// move oldpath to newpath
	result = renameat(AT_FDCWD, oldpath, AT_FDCWD, newpath);
	if (result != 0)
		return result;
	// move buffer to oldpath
	result = renameat(AT_FDCWD, buffer, AT_FDCWD, oldpath);
	return result;
#endif
}

#include <sys/file.h>  // flock
#include <fcntl.h>	   // open
#include <unistd.h>	   // close

// Try to get a lock. Returns its file descriptor or -1 if failed.
static int tryLockFile(const char* lockpath)
{
	// temporarily set umask to 0 to ensure that the file is created with
	// write permissions for the owner
	mode_t m = umask(0);
	int fd = open(lockpath, O_RDWR | O_CREAT, 0644);
	umask(m);

	// check the fd and call flock in exclusive and non-blocking mode
	if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
		close(fd);
		fd = -1;
	}

	return fd;
}

// Release the lock obtained with `tryLockFile(lockName)`.
static void releaseLock(int fd)
{
	if (fd < 0)
		return;

	// Note: Just closing the file descriptor releases the lock. Using the
	// LOCK_UN operation is not necessary and may lead to race conditions when
	// not handled correctly. Similarly, removing the lock file from disk may
	// lead to race conditions.
	close(fd);
}
