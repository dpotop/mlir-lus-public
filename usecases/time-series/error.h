#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

//===============================================================
// Error handling
#define FAILWITH(...) {					   \
    fprintf(stderr,"ERROR in %s, line %d, function %s:",   \
	    __FILE__, __LINE__, __func__) ;		   \
    fprintf(stderr,__VA_ARGS__) ;			   \
    fflush(stderr) ;					   \
    exit(0) ;						   \
  }

#define WARNING(...) {					   \
    fprintf(stderr,"WARNING in %s, line %d, function %s:", \
	    __FILE__, __LINE__, __func__) ;		   \
    fprintf(stderr,__VA_ARGS__) ;			   \
    fflush(stderr) ;					   \
  }

#define DEBUG_PRINTF(...) {				   \
    fprintf(stderr,"DEBUG in %s, line %d, function %s:",   \
	    __FILE__, __LINE__, __func__) ;		   \
    fprintf(stderr,__VA_ARGS__) ;			   \
    fflush(stderr) ;					   \
  }
