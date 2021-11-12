#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#include "../runtime/scheduler.h"
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

static int kbd_fd = -1 ;
void open_kbd() {
  kbd_fd = open("kbd",O_RDONLY|O_NONBLOCK) ;
  if(kbd_fd<0){
    perror("Cannot open named fifo.\n") ;
    //    exit(0) ;
  }
}


void sched_read_input_kbd(int32_t      input_id,
			  // Here start the memref struct
			  void*        allocated,
			  void*        aligned,
			  intptr_t     offset,
			  ...) {
  char* tmp_buf = (char*)aligned ;
  read(kbd_fd,tmp_buf,1);
}

void sched_read_input_snd(int32_t      input_id,
			  // Here start the memref struct
			  void*        allocated,
			  void*        aligned,
			  intptr_t     offset,
			  ...) {
  int r;
  int datasize = sizeof(int16_t);
  int shape = 512;
  int r_acc = datasize * shape;
  char* tmp_buf = (char*)aligned ;
  for(;r_acc;r_acc-=r,tmp_buf+=r) {
    r = read(0,tmp_buf,r_acc) ;
    if(r<0) {
      perror("read_samples error:") ;
      exit(0) ;
    } 
  }
}

int32_t sched_write_output_snd(int32_t      output_id,
			       // Here start the memref struct
			       void*        allocated,
			       void*        aligned,
			       intptr_t     offset,
			       ...) {
  int w;
  int datasize = sizeof(int16_t);
  int shape = 512;
  int w_acc = datasize * shape;
  char* tmp_buf = (char*)aligned ;
    for(;w_acc;w_acc-=w,tmp_buf+=w) {
    w = write(1,tmp_buf,w_acc) ;
    if(w<0) {
      perror("write_samples error:") ;
      exit(0) ;
    }
  }
  fflush(stdout) ;
  return 0;
}

void pitch(TASK_ID_TYPE tid,
	   void    (*f1)(int32_t, void*, void*, intptr_t, ...),
	   void    (*f2)(int32_t, void*, void*, intptr_t, ...),
	   int32_t (*f3)(int32_t, void*, void*, intptr_t, ...));

void task1_start(TASK_ID_TYPE tid) {
  setup_time_logging(0);
  open_kbd();
  pitch(tid, sched_read_input_kbd, sched_read_input_snd,
	sched_write_output_snd);
}

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
