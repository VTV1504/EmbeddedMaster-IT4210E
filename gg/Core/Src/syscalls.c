/*
 * syscalls.c
 *
 *  Created on: Jun 21, 2025
 *      Author: Nam
 */


#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

int _write(int file, char *ptr, int len) {
    return len;
}

int _read(int file, char *ptr, int len) {
    errno = ENOSYS;
    return -1;
}

int _close(int file) {
    errno = ENOSYS;
    return -1;
}

int _lseek(int file, int ptr, int dir) {
    errno = ENOSYS;
    return -1;
}

int _fstat(int file, struct stat *st) {
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file) {
    return 1;
}

int _kill(int pid, int sig) {
  return -1;
}

int _getpid(void) {
  return 1;
}
