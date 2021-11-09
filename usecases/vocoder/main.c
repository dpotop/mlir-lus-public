#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <stdint.h>

#include "error.h"
#include "scheduler.h"
#include "memrefs.h"

void sched_set_output_memrefxi16(TASK_ID_TYPE tid,
				  int32_t      input_id,
				  int32_t      dimensions,
				  int32_t      datasize,
				  // Here start the memref struct
				  void*        allocated,
				  void*        aligned,
				  intptr_t     offset,
				 ...);

void sched_set_input_memrefxi8(TASK_ID_TYPE tid,
			       int32_t      input_id,
			       int32_t      dimensions,
			       int32_t      datasize,
			       // Here start the memref struct
			       void*        allocated,
			       void*        aligned,
			       intptr_t     offset,
			       ...);

void sched_set_input_memrefxi16(TASK_ID_TYPE tid,
			       int32_t      input_id,
			       int32_t      dimensions,
			       int32_t      datasize,
			       // Here start the memref struct
			       void*        allocated,
			       void*        aligned,
			       intptr_t     offset,
				...);

void task1_start(TASK_ID_TYPE tid);

// This is the main task of the system, which instantiates the
// tick synchronization and instantiates the main node. It should
// be seen as part of the system/sched, not of the application.
void tick_task(TASK_ID_TYPE tid) {
  for(;;) {

    struct memref_1d_i8 kbd;
    struct memref_1d_i16 sndbufin;
    struct memref_1d_i16 product;
    alloc_memref_1d_i8(&kbd, 1);
    alloc_memref_1d_i16(&sndbufin, 512);
    alloc_memref_1d_i16(&product, 512);
    
    int8_t *kbd_aligned = (int8_t*)kbd.aligned;
    for(int i=0;i<1;i++)kbd_aligned[i] = i ;
    int16_t *sndbufin_aligned = (int16_t*)sndbufin.aligned;
    for(int i=0;i<512;i++)sndbufin_aligned[i] = i ;

    bzero_memref_1d_i16(&product) ;
    
    sch_set_instance(1,task1_start,2,1) ;
    sched_set_input_memrefxi8(1,0,1,sizeof(int8_t),
			      kbd.allocated,
			      kbd.aligned,
			      kbd.offset,
			      kbd.size[0],
			      kbd.stride[0]);
    sched_set_input_memrefxi16(1,1,1,sizeof(int16_t),
			       sndbufin.allocated,
			       sndbufin.aligned,
			       sndbufin.offset,
			       sndbufin.size[0],
			       sndbufin.stride[0]);
    sched_set_output_memrefxi16(1,0,1,sizeof(int16_t),
				product.allocated,
				product.aligned,
				product.offset,
				product.size[0],
				product.stride[0]);
    inst(1) ;
    /* print_memref_2d_i32(&to_task) ; */
    /* print_memref_2d_i32(&from_task) ; */
    /* free_memref_2d_i32(&to_task) ; */
    /* free_memref_2d_i32(&from_task) ; */
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
