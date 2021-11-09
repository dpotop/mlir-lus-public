// -*- C++ -*- //

#ifndef INLINE_NODES_H
#define INLINE_NODES_H
#include "mlir/IR/BuiltinOps.h"
#include <functional>
#include <list>
#include <unordered_set>
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/Instance.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct InlineNodes : public unary_function < ModuleOp, void > {
    public:
      void operator() (ModuleOp moduleOp);
    private:
      unordered_set<Operation*> toDo;
      unordered_set<Operation*> ready;
      unordered_set<Operation*> done;
      unordered_set<Operation*> dishes;

      void packToDoNodes(ModuleOp);

      bool containsInstance(NodeOp);

      void packReadyNodes();

      void inlineReadyNodes(ModuleOp);

      void inlineNode(NodeOp, InstanceOp);
    };
  }
}

#endif
