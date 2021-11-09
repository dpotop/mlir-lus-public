//  -*- C++ -*- //

#ifndef CLOCK_NEW_H
#define CLOCK_NEW_H


#include <mlir/IR/Value.h> // For the isa<> macros
#include "TestCondition.h"

namespace mlir {
  namespace lus {

    //============================================================
    // Clocks

    // Pre-declaration, needed in Clock
    class FreeClock ;
    class ClockRepository ;
    
    // 
    class Clock {
    public:
      // The kind of the clock
      enum Kind {
	CkBase,   // Base clock
	CkFree,   // Free clock
	CkWhen,   // When clock with base clock
      };
    private:
      const Kind kind;
    public:
      Kind getKind() const { return kind; }
      
    private:
      Clock();
    public:
      // Basic stuff
      virtual bool operator==(const Clock& c) const = 0;
      bool operator!=(const Clock& c) const { return !operator==(c) ; }
      
      // Printing method, which is a call to stream, below
      friend raw_ostream& operator<<(raw_ostream& os, Clock& c) ;
    protected:
      // Only clock analysis should create clocks
      Clock(Kind k) : kind(k) {}
      virtual raw_ostream& stream(raw_ostream& os) = 0;
      friend class ClockRepository ;
    public:
      virtual ~Clock() {}
      virtual Clock* clone() const = 0 ;
      virtual void debugPrint(raw_ostream& os) = 0;
	
    };

    inline raw_ostream& operator<<(raw_ostream& os, Clock& c) {
      return c.stream(os) ;
    }    

    class BaseClock : public Clock {
      // Hide copy constructor
      BaseClock(const BaseClock& c) ;
    protected:
      // Only the ClockRepository class should be able to create
      // such objects.
      BaseClock() : Clock(CkBase) {}
      friend class ClockRepository ;

    public:
      ~BaseClock() {}
      virtual Clock* clone() const override { return new BaseClock() ; }
      bool operator==(const Clock& c) const override {
	return (c.getKind() == Clock::CkBase);
      }

      static bool classof(const Clock* c) { return c->getKind() == CkBase ; }
      // Support for clock analysis
    protected:
      raw_ostream& stream(raw_ostream& os) override {
	return (os << ".") ;
      }
    public:
      virtual void debugPrint(raw_ostream& os) override {
	stream(os) ;
      }
    };

    class FreeClock : public Clock {
      unsigned code;
      static unsigned newCode;
    public:
      inline const unsigned getCode() const { return code ; }

    private:
      // Hide copy constructor
      FreeClock(const FreeClock& c)
	: Clock(c.getKind()),code(c.code) {}
      // Only the ClockRepository class should be able to create
      // such objects.
      FreeClock() : Clock(CkFree), code(newCode++) {}
      friend class ClockRepository ;
    public:
      ~FreeClock() {}
      virtual Clock* clone() const override {
	return new FreeClock(*this) ;
      }

      
    public:
      bool operator==(const Clock& c) const override {
	if(isa<FreeClock>(&c)) {
	  const FreeClock& cc = cast<FreeClock>(c) ;
	  return (code==cc.code) ;
	} else return false ;
      }
      
      static bool classof(const Clock* c) { return c->getKind() == CkFree ; }
      // Support for clock analysis
    protected:
      raw_ostream& stream(raw_ostream& os) override {
	return (os << "CkFree<" << code << ">") ;
      }
    public:
      virtual void debugPrint(raw_ostream& os) override {
	stream(os) ;
      }
      
    };

    class WhenClock : public Clock {
      // Base clock - I use a pointer here, as if I use a ref
      // it does not allow me to change it (as if it was constant).
      Clock* clk;
      // Condition
      const Cond<Value> condi;
    public:
      inline const Cond<Value> cond() const { return condi; }
      inline Clock& clock() const { return *clk ; }
      
    private:
      // Hide some constructors
      WhenClock();
      WhenClock(const WhenClock& c)
	: Clock(c.getKind()),
	  clk(c.clk->clone()),
	  condi(c.condi) {}
      // Only the ClockRepository class should be able to create
      // such objects.
      WhenClock(Clock& cl, const Cond<Value>& cond) :
        Clock(CkWhen),
	clk(cl.clone()),
	condi(cond) {}
      friend class ClockRepository ;
    public:
      virtual ~WhenClock() { delete clk ; }
      virtual Clock* clone() const override {
	return new WhenClock(*this) ;
      }
      // Get the base clock of the When condition hierarchy
      virtual const Clock& getBaseClock() const {
	if(isa<WhenClock>(clk)) {
	  const WhenClock& wck = cast<WhenClock>(clock()) ;
	  return wck.getBaseClock() ;
	} else return clock() ;
      }

    private:
      // Only the ClockRepository class should be able to
      // perform substitution
      virtual void substitute(FreeClock& oldClock,
			      Clock& newClock) {
	if((*clk) == oldClock) {
	  // delete clk ;
	  clk = newClock.clone() ;
	} else {
	  if(clk->getKind() == CkWhen) {
	    // Even if it's not identical, if this is a WhenClock,
	    // I still have to traverse it to determine if clock1 is
	    // not used as a base clock.
	    WhenClock* wcn = cast<WhenClock>(clk) ;
	    wcn->substitute(oldClock,newClock) ;
	  }
	}
      }

    public:
      bool operator== (const Clock& c) const override {
	if(isa<WhenClock>(&c)) {
	  const WhenClock&cc = cast<WhenClock>(c) ;
	  return
	    (getKind() == cc.getKind())
	    &&(cond() == cc.cond())
	    &&(clock() == cc.clock()) ;
	} else return false ;
      }

      static bool classof(const Clock* c) { return (c->getKind() == CkWhen) ; }
    private:
      raw_ostream& stream(raw_ostream& os) override {
	os << clock() << " " ;
	os << "when"  << " " ;
	cond().print(os) ;
	os << " " ;
	return os ;
      }
    public:
      virtual void debugPrint(raw_ostream& os) override {
	clk->debugPrint(os) ;
	os << " when " ;
	cond().debugPrint(os) ;
	os << " " ;
      }
    };




    
  }
}

#endif
