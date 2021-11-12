#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>

//================================================================
// Task identifier type
typedef int32_t TASK_ID_TYPE ;
// A task identifier representing an inexistent task
#define NO_TASK  INT32_MAX
// Tasks can have identifiers between 0 and (MAX_TASKS-1).
// Task 0 is always the "tick" task.
#define MAX_TASKS 32 

//================================================================
// Task type
typedef void (*TASK_TYPE)(TASK_ID_TYPE) ;

//================================================================
// Scheduler initialization and start of the scheduling
void          sched_init(void);
__attribute__((noreturn))
void          sched_start(void);

//================================================================
// Task operations

// Task creation requires giving a task identifier
// between 0 and (MAX_TASKS-1).
void          sched_create_task(TASK_ID_TYPE id,TASK_TYPE task,int inputs,int outputs);
void          sched_relinquish(void);
void          sched_set_task(TASK_ID_TYPE task_id) ;
int           sched_task_exists(TASK_ID_TYPE task_id) ;
void          sched_task_check(TASK_ID_TYPE task_id, TASK_TYPE task) ;
TASK_ID_TYPE  sched_get_tid(void) ;
TASK_ID_TYPE  sched_get_parent_tid(void) ;

//================================================================
// Moving up the abstraction levels
void setup_time_logging(bool log_time) ;
int32_t tick(void) ;
void inst(TASK_ID_TYPE id) ;
void sch_set_instance(TASK_ID_TYPE id,TASK_TYPE start_func,int32_t inputs,int32_t outputs) ;

//============================================================
// Task communication - description is provided in scheduler.c
// Attention, for each task:
// - inputs must be numbered from 0 to number_of_inputs-1
// - outputs must be numbered from 0 to number_of_outputs-1

// Functions to call on the instantiator side
void sched_set_input(TASK_ID_TYPE tid,
		     int32_t      input_id,
		     int32_t      dimensions,
		     int32_t      datasize,
		     // Here start the memref struct
		     void*        allocated,
		     void*        aligned,
		     intptr_t     offset,
		     va_list      listptr) ;
void sched_set_output(TASK_ID_TYPE tid,
		      int32_t      output_id,
		      int32_t      dimensions,
		      int32_t      datasize,
		      // Here start the memref struct
		      void*        allocated,
		      void*        aligned,
		      intptr_t     offset,
		      va_list      listptr) ;

// Functions to call on the instantiated side
void sched_read_input(int32_t      input_id,
		      // Here start the memref struct
		      void*        allocated,
		      void*        aligned,
		      intptr_t     offset,
		      ...) ;
int32_t sched_write_output(int32_t      output_id,
			   // Here start the memref struct
			   void*        allocated,
			   void*        aligned,
			   intptr_t     offset,
			   ...) ;

