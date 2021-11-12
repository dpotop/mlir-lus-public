#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

static unsigned long start = 0;
static unsigned long start_usec = 0;

void init_time() {
  struct timeval time;
  gettimeofday(&time, NULL);
  start = time.tv_sec;
  start_usec = time.tv_usec;
}

void tick() {
  struct timeval time;
  gettimeofday(&time, NULL);
  unsigned long stop = time.tv_sec;
  unsigned long stop_usec = time.tv_usec;
  unsigned long elapsed = (stop - start) * 1000000 + stop_usec - start_usec;
  printf("tick: %lu us\n", elapsed);
  start = stop;
  start_usec = stop_usec;
}

void halt() {}
