#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>

#include "error.h"
#include "scheduler.h"

// Redefine the macro to shut off debug messages
/* #define DEBUG_PRINTF(...) */

//===============================================================
// Scheduler constants
//
// Stack size allocated to each task/process
#define TASK_STACK_SIZE (128*1024)
// Max numbers of inputs and outputs for each task
#define MAX_TASK_INPUTS 16
#define MAX_TASK_OUTPUTS 16


//===============================================================
// Internal scheduler state

//---------------------------------------------------------------
// Communication buffers
// The communication buffers are patterned on MLIRs memrefs.
// The memory organization of a memref with N dimensions and 
// data of type T is the following:
// struct memref {
//   void*    allocd ;   // Allocated memory, which must be freed upon deallocation
//   T*       aligned ;  // Aligned memory where data sits (in practice always equal
//                       // to allocd).
//   intptr_t offset ;   // Offset where the data is found (in practice always 0).
//   intptr_t size[N] ;  // The data size on the N dimensions
//   intptr_t stride[N] ;// The stride on the N dimensions
// } ;
// Note that the struct is expressible in C only if the number of
// dimensions is fixed. For this reason, to avoid creating large numbers
// of kernel functions, we will use a generic struct, and will always
// pass the number of dimensions in argument to the manipulation functions.

// This struct maintains a fixed-form of this struct meant to
// represent memrefs of a maximum dimensionality.
#define MAX_MEMREF_DIM 6
struct memref {
  // Not present in the original memref
  int      dimensions ; // A value of -1 means it is not initialized
  int      datasize ;   // This should contain the sizeof of the data type
  // Present in the original memref
  void*    allocd ;
  void*    aligned ;
  intptr_t offset ;
  intptr_t shape[2*MAX_MEMREF_DIM] ;
} ;


//---------------------------------------------------------------
// Task records
struct task {
  //-------------------------------------------------------
  // Fixed task characteristics
  
  // Some are provided as arguments of create_task
  TASK_TYPE func; // Task function - no input and no output
  int stack_size; // Stack size

  int number_of_inputs ;
  struct memref* inputs ;
  int number_of_outputs ;
  struct memref* outputs ;
  
  // Some are computed and fixed during execution of create_task
  void *stack_bottom;  // Stack bottom
  void *stack_top;     // Stack top
  TASK_ID_TYPE parent_tid ; // The task id of the parent (if any)

  //-------------------------------------------------------
  // Variable task characteristics
  
  // Task status
  enum {
    ST_UNUSED,    // Task identifier/record unused
    ST_CREATED, // Task created, but not yet started
    ST_RUNNING, // Task running
  } status;

  // Saved program counter (only meaningful in a process/task
  // that's beed already started).
  jmp_buf program_counter;

};


//---------------------------------------------------------------
// Internal scheduler state

// Task identifier pointing to no task (sort of NULL for task
// identifiers).
struct sched_state {
  // Static allocation of all possible task records (all initially
  // empty). Task identifiers are indices in this vector of task
  // records.
  struct task task_list[MAX_TASKS] ;

  // The task currently running, for which the state must be saved
  // during a context switch.
  TASK_ID_TYPE running_task_id ;
  // The next task to run, for which the state must be loaded during
  // a context switch.
  TASK_ID_TYPE next_task_id ;

  // Entry point for the scheduler request handler (a form
  // of soft interrupt handler).
  jmp_buf handler;

} sched_state;

//===============================================================
// Scheduler routines

//---------------------------------------------------------------
// Initialize the internal scheduler state. No task created, no
// task running.
void sched_init(void)
{
  // Mark all task records as unused
  for(TASK_ID_TYPE i=0;i<MAX_TASKS;i++)
    sched_state.task_list[i].status = ST_UNUSED ;
  sched_state.running_task_id = NO_TASK ;
  sched_state.next_task_id = NO_TASK ;
}

//---------------------------------------------------------------
// Determine if the task has been created
int sched_task_exists(TASK_ID_TYPE task_id) {
  if(task_id < 0)
    FAILWITH("Malformed (negative) task identifier.") ;
  if(task_id >= MAX_TASKS)
    FAILWITH("Malformed task identifier, bigger than MAX_TASKS (%d).",MAX_TASKS) ;
  return (sched_state.task_list[task_id].status != ST_UNUSED) ;
}
// Check if the task of a created function is the correct one.
void sched_task_check(TASK_ID_TYPE task_id, TASK_TYPE func) {
  if(!sched_task_exists(task_id)) FAILWITH("Task does not exist.") ;
  if(func != sched_state.task_list[task_id].func)
    FAILWITH("Task does not have the correct function.") ;
  return ;
}


//---------------------------------------------------------------
// Create a task. Only needs as argument the task function. The
// identifier is fixed. Returns the task identifier (unless it fails,
// in which case it terminates execution).
void sched_create_task(TASK_ID_TYPE id,TASK_TYPE func,int inputs,int outputs) {
  if(sched_task_exists(id))
    FAILWITH("Trying to create the same task twice.") ;  
  struct task *task = &sched_state.task_list[id] ;
  task->status = ST_CREATED;
  task->func = func;
  task->stack_size = TASK_STACK_SIZE;
  // For now, malloc works. If alignment errors occur, use aligned_alloc
  task->stack_bottom = malloc(task->stack_size); 
  task->stack_top = task->stack_bottom + task->stack_size;
  // Finally, set up the parent tid
  task->parent_tid = sched_state.running_task_id ;
  // Setup place for inputs and outputs
  if(inputs<0) FAILWITH("Negative number of inputs.") ;
  if(outputs<0) FAILWITH("Negative number of inputs.") ;
  task->inputs = (struct memref*)malloc(inputs*sizeof(struct memref)) ;
  task->outputs = (struct memref*)malloc(outputs*sizeof(struct memref)) ;
  task->number_of_inputs = inputs ;
  task->number_of_outputs = outputs ;
  for(int i=0;i<inputs;i++)
    task->inputs[i].dimensions = -1 ;
  for(int i=0;i<outputs;i++)
    task->outputs[i].dimensions = -1 ;
}

//---------------------------------------------------------------
// Set the next task instance to execute
void sched_set_task(TASK_ID_TYPE task_id) {
  if(sched_task_exists(task_id)) {
    // Set it to be the next task to execute after a context switch
    sched_state.next_task_id = task_id ;
  } else {
    FAILWITH("sched_set_task: No such task %d.\n",task_id) ;
  }
}


//---------------------------------------------------------------
// 
__attribute__((noreturn))
static void schedule(void) {
  if(sched_state.next_task_id == NO_TASK) {
    FAILWITH("Attempting to start sched without creating at least one task.") ;
  }
  TASK_ID_TYPE tid = sched_state.next_task_id ;
  if (sched_state.task_list[tid].status == ST_CREATED) {
    // This task has not been started yet. Assign a new stack,
    // reset the stack pointer, and run the task, which does 
    // not complete.
    volatile register void *top = sched_state.task_list[tid].stack_top;
    // Set the stack for the task and run the function
    asm volatile("mov %[rs], %%rsp \n" : [ rs ] "+r" (top) ::);
    sched_state.task_list[tid].status = ST_RUNNING;
    sched_state.running_task_id = tid ;
    sched_state.task_list[tid].func(tid);
    FAILWITH("Task function terminated execution.") ;
  } else {
    sched_state.running_task_id = tid ;
    longjmp(sched_state.task_list[tid].program_counter,
	    1 /* This value must be !=0 */ ) ;
  }
}



//---------------------------------------------------------------
// This routine starts the scheduler

// When control is given to the scheduler, we use an integer code
// to determine what the request is.
enum {
  INIT=0,     // The initialization call to scheduler_run. This
              // has to be 0 to correspond to the 0 return code
              // of setjmp.
  RELINQUISH, // Relinquish request from one of the tasks.
};

// This routine is called exactly once in the program entry
// point to start the scheduler. It works by setting up a
// context to which all tasks can relinquish control. The
// context remains unchanged between the different calls.
// Note that this approach will probably *not* work correctly
// in multi-threaded contexts, because it assumes only one
// instance of the scheduler can run at any given time.
void sched_start(void) {
  // Recall how setjmp/longjmp work. This routine is called by
  // the entry point to set up the scheduler request handler,
  // and in this case it returns 0.
  // DEBUG_PRINTF("sched_start: phase 0\n");
  int sched_request = setjmp(sched_state.handler) ;
  // DEBUG_PRINTF("sched_start: phase 1: request:%d\n",sched_request);
  switch (sched_request) {
  case INIT:
    schedule() ;
    FAILWITH("This point should be unreachable.") ;
  case RELINQUISH:
    schedule();
    FAILWITH("This point should be unreachable.") ;
  default:
    FAILWITH("Unknown sched request identifier.") ;
  }
}

//---------------------------------------------------------------
// This routine is called by the tasks when they are ready to
// relinquish control to the scheduler. It will be the
// implementation of the tick() primitive of mlirlus.
void sched_relinquish(void) {
  // The way setjmp/longjmp work is a bit counter-intuitive, because
  // the following call to setjmp will return more than once for
  // each call.
  // - The first time, it will return with a retcode==0, and it will
  //   store the task PC in the appropriate scheduler state field.
  //   In this case, it will call longjmp to give back control to
  //   the scheduler. Note the use of the RELINQUISH code passed
  //   to the scheduler. Other codes may be used here, allowing
  //   the creation of a soft interrupt mechanism (and so
  //   we need codes to disambiguate).
  // - The following times, it will return to the context that was
  //   saved. This happens when the scheduler gives back control
  //   to the task, and in this case the function scheduler_relinquish
  //   can terminate, allowing the calling task to continue its
  //   execution.
  // DEBUG_PRINTF("sched_relinquish(%d)\n",sched_state.running_task_id) ;
  int retcode =
    setjmp(sched_state.task_list[sched_state.running_task_id].program_counter) ;
  if (retcode == 0) {
    longjmp(sched_state.handler, RELINQUISH);
  } else {
    return;    
  }
}

//---------------------------------------------------------------
// Routines to get the task identifier of the running task and
// of the parent task of the running task (if any).
TASK_ID_TYPE sched_get_tid(void) {
  return sched_state.running_task_id ;
}
TASK_ID_TYPE sched_get_parent_tid(void) {
  return sched_state.task_list[sched_state.running_task_id].parent_tid ;
}

//======================================================================
// The two core routines for giving control between tasks:
// - tick() returns control to the instantiating module
// - inst() gives control for one tick to an instantiated module

// We implement in the tick routine a time logging mechanism.
// By default, this mechanism is inactive. It can be
// activated by a call to set_time_logging(true).
// After the point where this happens, each call to tick() will
// log to the standard error the time (in usec) that lapsed since
// the last call to tick() (or from the call to set_time_logging,
// for the first tick().
static bool log_time_flag = false ;
static struct timeval stored_timeval ;
void setup_time_logging(bool log_time) {
  log_time_flag = log_time ;
  if(log_time == true) {
    gettimeofday(&stored_timeval,NULL) ;
  }
}

int32_t tick(void) {
  if(log_time_flag) {
    TASK_ID_TYPE tid = sched_get_tid();
    if (tid == 1) {
      struct timeval t ;
      gettimeofday(&t, NULL);
      unsigned long long elapsed_usec =
	(t.tv_sec - stored_timeval.tv_sec)*1000000 +
	(t.tv_usec - stored_timeval.tv_usec) ;
      stored_timeval = t ;
      DEBUG_PRINTF("Tick duration: %llu\n",elapsed_usec) ;
    }
  }
  sched_set_task(sched_get_parent_tid()) ;
  sched_relinquish() ;
  return 0;
}

void inst(TASK_ID_TYPE id) {
  sched_set_task(id) ;
  sched_relinquish() ;
}

//======================================================================
// Move up in abstraction level. The sch_set_instance routine will
// check if the task exists and, if not, create it.
void sch_set_instance(TASK_ID_TYPE id,TASK_TYPE start_func,int32_t inputs,int32_t outputs) {
  if(sched_task_exists(id)) {
    // The task has already been created, but I still check the function
    sched_task_check(id, start_func) ;
  } else {
    sched_create_task(id,start_func,inputs,outputs) ;
  }
}


//===============================================================
// Communication between tasks (always between a parent task
// and its running child).

//---------------------------------------------------------------
// Set an input or output variable from the caller.
// It places the elements of the memref data structure in the
// internal structures of the runtime (but does not copy the
// data).
void sched_set_input(TASK_ID_TYPE tid,
		     int32_t      input_id,
		     int32_t      dimensions,
		     int32_t      datasize,
		     // Here start the memref struct
		     void*        allocated,
		     void*        aligned,
		     intptr_t     offset,
		     va_list listptr) {
  if(!sched_task_exists(tid)) FAILWITH("Inexistent task.") ;
  if((input_id<0)||(input_id>=sched_state.task_list[tid].number_of_inputs))
    FAILWITH("Malformed input identifier.") ;
  if((dimensions<0)||(dimensions>MAX_MEMREF_DIM))
    FAILWITH("Malformed dimensions number.") ;
  sched_state.task_list[tid].inputs[input_id].dimensions = dimensions ;
  sched_state.task_list[tid].inputs[input_id].datasize   = datasize ;
  sched_state.task_list[tid].inputs[input_id].allocd     = allocated ;
  sched_state.task_list[tid].inputs[input_id].aligned    = aligned ;
  sched_state.task_list[tid].inputs[input_id].offset     = offset ;
  if(offset != 0) FAILWITH("Memref offset should be zero.");
  for(int i=0;i<2*dimensions;i++) {
    intptr_t tmp = va_arg(listptr,intptr_t) ;
    sched_state.task_list[tid].inputs[input_id].shape[i] = tmp;
  }
}

void sched_set_output(TASK_ID_TYPE tid,
		      int32_t      output_id,
		      int32_t      dimensions,
		      int32_t      datasize,
		      // Here start the memref struct
		      void*        allocated,
		      void*        aligned,
		      intptr_t     offset,
		      va_list listptr) {
  if(!sched_task_exists(tid)) FAILWITH("Inexistent task.") ;
  if((output_id<0)||(output_id>=sched_state.task_list[tid].number_of_outputs))
    FAILWITH("Malformed output identifier.") ;
  if((dimensions<0)||(dimensions>MAX_MEMREF_DIM))
    FAILWITH("Malformed dimensions number.") ;
  sched_state.task_list[tid].outputs[output_id].dimensions = dimensions ;
  sched_state.task_list[tid].outputs[output_id].datasize   = datasize ;
  sched_state.task_list[tid].outputs[output_id].allocd     = allocated ;
  sched_state.task_list[tid].outputs[output_id].aligned    = aligned ;
  sched_state.task_list[tid].outputs[output_id].offset     = offset ;
  if(offset != 0) FAILWITH("Memref offset should be zero.");
  for(int i=0;i<2*dimensions;i++) {
    sched_state.task_list[tid].outputs[output_id].shape[i] = va_arg(listptr,intptr_t) ;
  }
}

//---------------------------------------------------------------
// Copy the input data onto the local variable, on the instantiated
// module side. This involves a full copy of the data, but not the
// transfer of the data size (which cannot be changed, given that
// allocation is done by the instantiated size).
// Even the copy should not be needed, but I do not know how to just
// give the data pointer with the current calling convention.
void sched_read_input(int32_t      input_id,
		      // Here start the memref struct
		      void*        allocated,
		      void*        aligned,
		      intptr_t     offset,
		      ...) {
  TASK_ID_TYPE tid = sched_get_tid() ;
  if((input_id<0)||(input_id>=sched_state.task_list[tid].number_of_inputs))
    FAILWITH("Malformed input identifier.") ;
  if(offset != 0) FAILWITH("Memref offset should be zero.");
  int size = sched_state.task_list[tid].inputs[input_id].datasize ;
  for(int i=0;i<sched_state.task_list[tid].inputs[input_id].dimensions;i++) {
    size *= sched_state.task_list[tid].inputs[input_id].shape[i] ;
  }
  memcpy(aligned,
	 sched_state.task_list[tid].inputs[input_id].aligned,
	 size) ;
}

//---------------------------------------------------------------
// Copy the local data onto the output variable, on the instantiated
// module side. This involves a full copy of the data, but not the
// transfer of the data size (which cannot be changed, given that
// allocation is done by the instantiated size).
// Even the copy should not be needed, but I do not know how to just
// give the data pointer with the current calling convention.
int32_t sched_write_output(int32_t      output_id,
			// Here start the memref struct
			void*        allocated,
			void*        aligned,
			intptr_t     offset,
			...) {
  TASK_ID_TYPE tid = sched_get_tid() ;
  if((output_id<0)||(output_id>=sched_state.task_list[tid].number_of_outputs))
    FAILWITH("Malformed output identifier.") ;
  if(offset != 0) FAILWITH("Memref offset should be zero.");
  int size = sched_state.task_list[tid].outputs[output_id].datasize ;
  for(int i=0;i<sched_state.task_list[tid].outputs[output_id].dimensions;i++) {
    size *= sched_state.task_list[tid].outputs[output_id].shape[i] ;
  }
  memcpy(sched_state.task_list[tid].outputs[output_id].aligned,
	 aligned,
	 size) ;
  return 0;
}
