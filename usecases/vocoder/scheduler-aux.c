#include "scheduler.h"
#include "error.h"
#include <stdio.h>
#include <stdarg.h>

void task1_start(TASK_ID_TYPE tid) {
  open_kbd();
  pitch(tid, sched_read_input_kbd, sched_read_input_snd,
	sched_write_output_snd);
}

int32_t sched_write_output_memrefxf32(int32_t      output_id,
				      // Here start the memref struct
				      void*        allocated,
				       void*        aligned,
				      intptr_t     offset,
				      ...) {
  return sched_write_output(output_id, allocated, aligned, offset);
}

int32_t sched_write_output_memrefxi1(int32_t      output_id,
				     // Here start the memref struct
				     void*        allocated,
				     void*        aligned,
				     intptr_t     offset,
				     ...) {
  return sched_write_output(output_id, allocated, aligned, offset);
}

void sched_read_input_memrefxf32(int32_t      input_id,
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

void sched_read_input_memrefxi8(int32_t      input_id,
				// Here start the memref struct
				void*        allocated,
				void*        aligned,
				intptr_t     offset,
				...) {
  sched_read_input(input_id, allocated, aligned, offset);
}

void sched_set_input_memrefxf32(TASK_ID_TYPE tid,
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

void sched_set_input_memrefxi8(TASK_ID_TYPE tid,
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

void sched_set_input_memrefxi16(TASK_ID_TYPE tid,
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

void sched_set_output_memrefxf32(TASK_ID_TYPE tid,
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

void sched_set_output_memrefxi1(TASK_ID_TYPE tid,
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

void sched_set_output_memrefxi16(TASK_ID_TYPE tid,
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
