#include "scheduler.h"
#include "error.h"
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

static unsigned long start = 0;
static unsigned long start_usec = 0;

void init_time() {
  struct timeval time;
  gettimeofday(&time, NULL);
  start = time.tv_sec;
  start_usec = time.tv_usec;
}

int32_t tick(void) {
  TASK_ID_TYPE tid = sched_get_tid();
  if (tid == 1) {
    struct timeval time;
    gettimeofday(&time, NULL);
    unsigned long stop = time.tv_sec;
    unsigned long stop_usec = time.tv_usec;
    unsigned long elapsed = (stop - start) * 1000000 + stop_usec - start_usec;
    printf("timestamp: %lu us\n", elapsed);
    gettimeofday(&time, NULL);
    start = time.tv_sec;
    start_usec = time.tv_usec;
  }
  sched_set_task(sched_get_parent_tid()) ;
  sched_relinquish() ;
  return 0;
}


void task1_start(TASK_ID_TYPE tid) {
  if (tid == 1) {
    init_time();
  }
  timeseries(tid, sched_read_input, sched_write_output);
}

int32_t sched_write_output_memrefxxf32(int32_t      output_id,
				       // Here start the memref struct
				       void*        allocated,
				       void*        aligned,
				       intptr_t     offset,
				       ...) {
  return sched_write_output(output_id, allocated, aligned, offset);
}

void sched_read_input_memrefxxf32(int32_t      input_id,
				  // Here start the memref struct
				  void*        allocated,
				  void*        aligned,
				  intptr_t     offset,
				  ...) {
  sched_read_input(input_id, allocated, aligned, offset);
}

void sched_read_input_memrefxi1(int32_t      input_id,
				// Here start the memref struct
				void*        allocated,
				void*        aligned,
				intptr_t     offset,
				...) {
  sched_read_input(input_id, allocated, aligned, offset);
}

void sched_set_input_memrefxxf32(TASK_ID_TYPE tid,
				 int32_t      input_id,
				 int32_t      dimensions,
				 int32_t      datasize,
				 // Here start the memref struct
				 void*        allocated,
				 void*        aligned,
				 intptr_t     offset,
				 ...) {
  va_list args;
  va_start(args, offset);
  sched_set_input(tid, input_id, dimensions, datasize,
		  allocated, aligned, offset, args);
}

void sched_set_input_memrefxi1(TASK_ID_TYPE tid,
			       int32_t      input_id,
			       int32_t      dimensions,
			       int32_t      datasize,
			       // Here start the memref struct
			       void*        allocated,
			       void*        aligned,
			       intptr_t     offset,
			       ...) {
  va_list args;
  va_start(args, offset);
  sched_set_input(tid, input_id, dimensions, datasize,
		  allocated, aligned, offset, args);
}

void sched_set_input_memrefxxi32(TASK_ID_TYPE tid,
				 int32_t      input_id,
				 int32_t      dimensions,
				 int32_t      datasize,
				 // Here start the memref struct
				 void*        allocated,
				 void*        aligned,
				 intptr_t     offset,
				 ...) {
  va_list args;
  va_start(args, offset);
  sched_set_input(tid, input_id, dimensions, datasize,
		  allocated, aligned, offset, args);
}

void sched_set_output_memrefxxf32(TASK_ID_TYPE tid,
				  int32_t      input_id,
				  int32_t      dimensions,
				  int32_t      datasize,
				  // Here start the memref struct
				  void*        allocated,
				  void*        aligned,
				  intptr_t     offset,
				  ...) {
  va_list args;
  va_start(args, offset);
  sched_set_output(tid, input_id, dimensions, datasize,
		   allocated, aligned, offset, args);
}
