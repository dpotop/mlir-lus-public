#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <stdint.h>

#include "../runtime/scheduler.h"
#include "memrefs.h"

void sched_set_input_memrefxxf32(TASK_ID_TYPE tid,
				 int32_t      input_id,
				 int32_t      dimensions,
				 int32_t      datasize,
				 // Here start the memref struct
				 void*        allocated,
				 void*        aligned,
				 intptr_t     offset,
				 ...);

void sched_set_output_memrefxxf32(TASK_ID_TYPE tid,
				  int32_t      input_id,
				  int32_t      dimensions,
				  int32_t      datasize,
				  // Here start the memref struct
				  void*        allocated,
				  void*        aligned,
				  intptr_t     offset,
				  ...);

void timeseries(TASK_ID_TYPE tid,
		void    (*f1)(int32_t, void*, void*, intptr_t, ...),
		int32_t (*f2)(int32_t, void*, void*, intptr_t, ...));

void task1_start(TASK_ID_TYPE tid) {
  if (tid == 1) {
    setup_time_logging(1);
  }
  timeseries(tid, sched_read_input, sched_write_output);
}

// This is the main task of the system, which instantiates the
// tick synchronization and instantiates the main node. It should
// be seen as part of the system/sched, not of the application.
void tick_task(TASK_ID_TYPE tid) {
  for(;;) {
    struct memref_2d_f32 to_task, from_task ;
    alloc_memref_2d_f32(&to_task,3,1) ;
    alloc_memref_2d_f32(&from_task,3,1) ;
    float* aligned = (float*)to_task.aligned ;
    for(int i=0;i<3;i++)aligned[i] = (float)i ;
    bzero_memref_2d_f32(&from_task) ;
    
    sch_set_instance(1,task1_start,1,1) ;
    sched_set_input_memrefxxf32(1,0,2,sizeof(float),
				to_task.allocated,
				to_task.aligned,
				to_task.offset,
				to_task.size[0],
				to_task.size[1],
				to_task.stride[0],
				to_task.stride[1]) ;
    sched_set_output_memrefxxf32(1,0,2,sizeof(float),
				 from_task.allocated,
				 from_task.aligned,
				 from_task.offset,
				 from_task.size[0],
				 from_task.size[1],
				 from_task.stride[0],
				 from_task.stride[1]) ;
    inst(1) ;

    /* print_memref_2d_f32(&to_task) ; */
    /* print_memref_2d_f32(&from_task) ; */
    /* free_memref_2d_f32(&to_task) ; */
    /* free_memref_2d_f32(&from_task) ; */
  }
}

// 
int main(void) {
  // Initialize the scheduler
  sched_init() ;
  // Create the tasks
  sched_create_task(0,tick_task,0,0) ;
  // Start scheduling
  sched_set_task(0) ;
  sched_start();
}
