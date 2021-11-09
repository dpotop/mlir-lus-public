#include <unistd.h>
void tick() {
  // This is a soft real-time implementation of tick.
  // The objective is here simply to reduce the pressure on
  // soundcard I/O polling by making the software wait on
  // time for the largest part of the computation cycle.
  //
  // To determine the wait duration, one must consider:
  // - the sampling frequency (44.1kHz = 44100 samples/second)
  // - the number of samples treated during one execution
  //   cycle (256 samples/cycle)
  // These two determine that the computation throughput is
  // of 44100/256 = 172 cycles/second. This throughput is
  // naturally ensured by the interaction between the application,
  // the OS, and the sound HW.
  //
  // The (average) duration of one execution cycle is therefore
  // 1/172 seconds = 5.81 ms. However, the software only spends
  // a fraction of this time doing computations. Most of the
  // time it will wait on the sound input, which potentially involves
  // continuous polling.
  // Our objective is to reduce the amount of polling by ensuring
  // that, for much of the execution cycle, the application simply
  // waits on time. We do this by using the usleep system service.
  //
  // The argument provided to usleep must always be smaller than
  // the cycle average duration of 5.81 ms. Otherwise, the system
  // is non-schedulable.
  // Given the complexity of the OSs on which this application
  // is executed, no good estimation method exists beyond testing.
  // On a Intel Core i7 quad-core running at 2.5GHz in 16Go of RAM
  // under MacOSX 10.15.3, 4ms is sometimes insufficient (depending on
  // the system load), and 3.5ms is a safe value. The value must be
  // tuned for the architecture. We set here a very safe value of
  // 2ms that should allow execution on most modern platforms.
  // 
  // Note that input to usleep is given in microseconds. Thus, to
  // specify a delay of 2 ms, one should give usleep 2000 as argument.
  // Also note that the argument of usleep is a minimum delay.
  usleep(100) ;
}

void halt() {}
