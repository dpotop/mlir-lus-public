// -*- C++ -*- //

#ifndef LOWER_INST_H
#define LOWER_INST_H

#include "../../Dialects/Lus/Instance.h"
#include "../../Dialects/Sync/InstOp.h"
#include "mlir/Pass/Pass.h" // For ModuleOp
#include <list>
#include <functional>

namespace mlir {
  namespace sync {
    struct LowerInst: public std::unary_function <ModuleOp, void> {
    private:
      using SyncInstOp = sync::InstOp;
      using LusInstOp = lus::InstanceOp;
      std::list<Operation*> dishes;
      static int64_t nextId;
      void lower(Operation *op);
    public:
      void operator() (ModuleOp moduleOp);
      static int64_t getId() { return nextId++; }
    };
  }
}

#endif
