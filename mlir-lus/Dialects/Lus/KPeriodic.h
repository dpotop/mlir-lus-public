// -*- C++ -*- //

#ifndef KPERIODIC_H
#define KPERIODIC_H

#include <vector>
#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream ;
using namespace std;

namespace mlir {
  namespace lus {

    // Determine if a boolean vector is the Boolean complement
    // of another
    inline bool isComplement(const std::vector<bool>&v1,
			     const std::vector<bool>&v2) {
      if(v1.size() != v2.size()) return false ;
      for(int i=0;i<v1.size();i++) { if(v1[i] == v2[i]) return false ; }
      return true ;
    }

    // Printing Boolean vectors
    inline raw_ostream& operator<<(raw_ostream&os,const std::vector<bool>&v) {
      for(int i=0;i<v.size();i++) { os << (v[i]?'1':'0') ; }
      return os ;
    }
    
    class KPeriodic {
      // K-periodic words formed of a prefix and a period, both being
      // Boolean words. Even if there is a "not" during parsing, it
      // is removed by flipping all the bits of the two words.
      // I can't use std::bitset, even though they are far more efficient,
      // because these must have fixed length...
      std::vector<bool> prefix ;
      std::vector<bool> period ;
      
    public:
      static const KPeriodic initWord ;
      static const KPeriodic preWord ;
      const std::vector<bool>& getPrefix() const { return prefix ; }
      const std::vector<bool>& getPeriod() const { return period ; }
      KPeriodic(const std::vector<bool>&pref,
		const std::vector<bool>&peri,
		bool sense = true) : prefix(pref), period(peri) {
	if(sense == false) {
	  // I have to flip all the bits of the prefix and period
	  this->prefix.flip() ;
	  this->period.flip() ;
	} 
      }
      KPeriodic buildComplement() const {
	KPeriodic result(getPrefix(),getPeriod(),false) ;
	return result ;
      }
      bool operator==(const KPeriodic&w) const {
	return (prefix == w.prefix) && (period == w.period) ;
      }
      bool operator!=(const KPeriodic&w) const {
	return !operator==(w) ;
      }
      bool isHeadTail() {
	return prefix.size() == 1 && period.size() == 1;
      }
      bool isComplement(const KPeriodic&w) const {
	return
	  ::mlir::lus::isComplement(this->prefix,w.prefix)
	  && ::mlir::lus::isComplement(this->period,w.period) ;
      }
      void print(raw_ostream&os) const {
	os << prefix << "(" << period << ")" ;
      }

      size_t hash_value() const {
	std::hash<std::vector<bool>> ptr_hash;
	return ptr_hash(prefix) ^ ptr_hash(period);
      }
      
      // Printing method, which is a call to stream, below
      friend raw_ostream& operator<<(raw_ostream& os, const KPeriodic& w) ;
    } ;

    inline raw_ostream& operator<<(raw_ostream& os, const KPeriodic& w) {
      w.print(os) ;
      return os ;
    }
    
    KPeriodic* parseKPeriodic(llvm::StringRef str) ;

    struct KPeriodicHash {
      template <class KPeriodic>
      size_t operator() (const KPeriodic& kp) const {
	return kp.hash_value();
      }
    };
  }
}

#endif
