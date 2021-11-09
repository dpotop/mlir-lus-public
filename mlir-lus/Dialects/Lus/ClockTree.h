// lus/ClockTree.h - ClockTree class definition -*- C++ -*- //

#ifndef CLOCK_TREE_NEW_H
#define CLOCK_TREE_NEW_H

#include "Clock.h"
#include <unordered_set>
#include <unordered_map> 
#include <list>

using namespace std;

namespace mlir {
  namespace lus {

    /// Nodes containing operations and Edges (conditions) leading to Nodes
    struct ClockTree {

    public:

      using Edge = Cond<Value>;
      using Path = list<Edge>;

      /// Needed by maps
      struct EdgeHash {
	template < class Edge >
	size_t operator() (const Edge& edge) const {
	  return edge.getHashValue();
	}
      };

    private:

      /// The edges leading to the deeper nodes
      unordered_map<Edge, ClockTree*, EdgeHash> children;
      /// The operation on the current node
      unordered_set<const Operation*> operations;
      
    public:
      
      ClockTree() {}
      ~ClockTree();

      /// Add this op within the tree among this clock 
      void add(const Operation*, Clock&);
      /// Add this op at the root of the tree
      void add(const Operation*);
      // Get the path of an operation in the tree
      Path path(const Operation*);
      /// Substitute a value by another in the edges of this tree and
      /// all its subtrees
      void substitute(const Value, const Value);
      /// Check if two operations live in the same node of the tree
      bool equal(const Operation*, const Operation*);
      /// Check if the second operation's clock is on a sublock of the first's
      bool greater(const Operation*, const Operation*);
      
    private:
      
      /// Auxiliary function of void substitute(const Value, const Value)
      void substitute(const Edge, const Value);
      /// Auxiliary function of Path path(Operation*)
      /// Fills the Path given by reference
      bool path(const Operation*, Path&);

    private:
      
      ClockTree(const ClockTree& ct) {}
      ClockTree(const ClockTree* ct) {}
      
    };
  }
}

#endif
