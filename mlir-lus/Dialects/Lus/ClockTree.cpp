#include "ClockTree.h"
#include "mlir/IR/Operation.h"
#include <algorithm>

namespace mlir {
  namespace lus {

    ClockTree::~ClockTree() {
      // Delete each subtree of the current tree
      for(pair<Edge, ClockTree*> child : children) {
      	ClockTree* t = get<1>(child);
      	delete t;
      }
    }

    void ClockTree::add(const Operation* op, Clock& clock) {

      // The clock is empty now, which means that the operation must be
      // stored at the current level.
      if (isa<BaseClock>(clock)) {
	add(op);
      }
      // The clock is not empty yet, which means that next tree(s) must be
      // visited before storing the operation.
      else if (isa<WhenClock>(clock)) {

	WhenClock* whenClock = dyn_cast<WhenClock>(&clock);
	Cond<Value> cond = whenClock->cond();
	ClockTree* nextTree;
	
	// Build the corresponding edge if not built already.
	// Fetch the next tree (node) which must be visited.
	Edge edge(cond); 
	if (children.count(edge)) {
	  nextTree = children[edge];
	}
	else {
	  nextTree = new ClockTree();
	  children[edge] = nextTree;
	}

	// Visit the next tree.
	nextTree->add(op, whenClock->clock());
      }
      else {
	assert(false);
      }
    }

    void ClockTree::add(const Operation* op) {
      operations.insert(op);
    }

    ClockTree::Path ClockTree::path(const Operation* op) {
      Path p;
      if (path(op, p)) {
	return p;
      }
      // If this operation does not exist in the tree : critical bug
      else {
	assert(false);
      }
    }

    bool ClockTree::path(const Operation* op, ClockTree::Path& path) {
      // If the operation does not exist in this node, try deeper
      if (operations.find(op) == operations.end()) {
	for (pair<Edge, ClockTree*> child : children) {
	  Edge edge = get<0>(child);
	  ClockTree* clockTree = get<1>(child);
	  path.push_back(edge);
	  // Something found among this path : return
	  if (clockTree->path(op, path)) {
	    return true;
	  }
	  // Nothing found among this path : try another path
	  path.pop_back();
	}
	// Nothing found at all
	return false;
      }
      else {
	return true;
      }
    }

    void ClockTree::substitute(const Value current, const Value replacement) {
      // Maybe "whenot current" exists
      Edge edge1(current, false);
      substitute(edge1, replacement);
      // Or maybe "when current"
      Edge edge2(current, true);
      substitute(edge2, replacement);
    }
    
    void ClockTree::substitute(Edge current, Value replacement) {
      // If current is an edge starting at this node, replace it with the
      // same flag but the replacement value
      if (children.count(current) > 0) {
	ClockTree* tree = children[current];
	Edge newEdge(replacement, current.getWhenotFlag());
	auto it = children.find(current);
	children.erase(it);
	children[newEdge] = tree;
	tree->substitute(current.getData(), replacement);
      }
      // Continue the quest on each subtree
      else {
	for (pair<Edge, ClockTree*> p : children) {
	  get<1>(p)->substitute(current.getData(), replacement);
	}
      }
    }

    bool ClockTree::equal(const Operation* op1, const Operation *op2) {
      return path(op1) == path(op2);
    }

    bool ClockTree::greater(const Operation* op1, const Operation* op2) {
      Path p1 = path(op1);
      Path p2 = path(op2);
      // Search p1 as a sublist of p2 (so p2 is a subclock of p1)
      return search(p2.begin(), p2.end(), p1.begin(), p1.end())	!= p2.end();
    }
    
    

    
    
    

    
  }
}
