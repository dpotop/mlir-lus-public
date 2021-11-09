#include "Clock.h"

namespace mlir {
  namespace lus {
    // Free clock creation counter, initialized at 0
    unsigned FreeClock::newCode = 0;
  }
}
