// -*- C++ -*- //
#ifndef TEST_CONDITION_H
#define TEST_CONDITION_H

#include <iostream>
#include "llvm/Support/raw_ostream.h"
// #include "mlir/IR/AsmState.h"
// #include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "KPeriodic.h"
#include "mlir/IR/Types.h"

namespace mlir {
  namespace lus {
    
    // Helper function used with clocks and clock analysis.
    // Pretty-prints a value on a raw_ostream.
    template<class TestObject>
    void debugPrintObj(TestObject&v,raw_ostream&err) ;    


    //============================================================
    // Conditions of when and merge operations. They are used in:
    // - The types of when and merge operations
    // - Clocks used in clock analysis
    // These are template classes because:
    // - When and Merge operation types only store variable types
    // - Clocks and operations also store values

    template<class TestObject> class Cond ;
    
    enum CondType {
      CondDataType,
      CondKPType,
      CondEmptyType,
      CondTombstoneType
    } ;

    template<class TestObject>
    class CondStorage {
      friend class Cond<TestObject> ;
    private:
      //
      CondType type ;
    public:
      // Accessors
      CondType getType() const { return type ; }
      bool isNormal() const { return (type == CondDataType)||(type == CondKPType) ; }

    private:
      CondStorage() ;
      CondStorage(CondStorage<TestObject>&src) ;
    protected:
      CondStorage(CondType type):type(type) {}
      
      virtual bool operator==(const CondStorage<TestObject>&c) const = 0 ;
      bool operator!=(const CondStorage<TestObject>&c) const { return !operator==(c) ; } 
      virtual bool isComplement(const CondStorage<TestObject>&c) const = 0 ;
      virtual CondStorage<TestObject>* buildComplement() const = 0 ;

    public:
      // Printing
      virtual void print(raw_ostream&os) const = 0 ;
      virtual void debugPrint(raw_ostream&os) = 0 ;
    } ;
    
    template<class TestObject>
    class EmptyCondStorage : public CondStorage<TestObject> {
      friend class Cond<TestObject> ;
      EmptyCondStorage(CondStorage<TestObject>&src) ;
      // Hidden. Only the friend class Cond can use it.
      EmptyCondStorage() : CondStorage<TestObject>(lus::CondEmptyType) {}     
    public:
      virtual bool operator==(const CondStorage<TestObject>&c) const {
	return (c.getType() == CondStorage<TestObject>::getType()) ;
      }
      virtual bool isComplement(const CondStorage<TestObject>&c) const {
	// Complement comparison should not be called on Empty
	assert(false) ;
      }
      virtual CondStorage<TestObject>* buildComplement() const {
	assert(false) ;
      }
      // Printing
      virtual void print(raw_ostream&os) const { os << "ERR(EmptyType)" ; }
      virtual void debugPrint(raw_ostream&os) { print(os) ; }
    } ;

    template<class TestObject>
    class TombstoneCondStorage : public CondStorage<TestObject> {
      TombstoneCondStorage(CondStorage<TestObject>&src) ;
      // Hidden. Only the friend class Cond can use it.
      TombstoneCondStorage() : CondStorage<TestObject>(CondTombstoneType) {} 
    public:
      virtual bool operator==(const CondStorage<TestObject>&c) const {
	return (c.getType() == CondStorage<TestObject>::getType()) ;
      }
      virtual bool isComplement(const CondStorage<TestObject>&c) const {
	// Complement comparison should not be called on Tombstone
	assert(false) ;
      }
      virtual CondStorage<TestObject>* buildComplement() const {
	assert(false) ;
      }
      // Printing
      virtual void print(raw_ostream&os) const { os << "ERR(TombstoneType)" ; }
      virtual void debugPrint(raw_ostream&os) { print(os) ; }
      friend class Cond<TestObject> ;
    } ;

    template<class TestObject>
    class DataCondStorage : public CondStorage<TestObject> {
      // Actual fields
      bool  whenotFlag ;
      // The tested value (or its type), if any
      TestObject value ;
    public:
      // Implicit copy constructor on both
      bool getWhenotFlag() const { return whenotFlag ; }
      TestObject getValue() const { return value ; }
      
    private:
      DataCondStorage(CondStorage<TestObject>&src) ;
      DataCondStorage() ;
    public:
      DataCondStorage(bool whenotFlagIn,TestObject valueIn)
	: CondStorage<TestObject>(CondDataType),
	  whenotFlag(whenotFlagIn),
	  value(valueIn) {} 
      virtual bool operator==(const CondStorage<TestObject>&c) const {
	if(c.getType() != CondStorage<TestObject>::getType()) return false ;
	const DataCondStorage<TestObject>& dc = static_cast<const DataCondStorage<TestObject>&>(c) ;
	return (whenotFlag == dc.whenotFlag)&&(value == dc.value);
      }
      virtual bool isComplement(const CondStorage<TestObject>&c) const {
	if(c.getType() != CondStorage<TestObject>::getType()) return false ;
	const DataCondStorage<TestObject>& dc = static_cast<const DataCondStorage<TestObject>&>(c) ;
	return (whenotFlag != dc.whenotFlag)&&(value == dc.value);
      }
      virtual CondStorage<TestObject>* buildComplement() const {
	DataCondStorage<TestObject>* obj = new DataCondStorage<TestObject>(!whenotFlag,value) ;
	CondStorage<TestObject>* res = static_cast<CondStorage<TestObject>*>(obj) ;
	return res ;
      }
      // Printing
      virtual void print(raw_ostream&os) const {
	if(whenotFlag) os << " not " ;
	os << value << " " ;
      }
      virtual void debugPrint(raw_ostream&os) {
	if(whenotFlag) os << " not " ;
	debugPrintObj<TestObject>(value,os) ;
	os << " " ;
      }
    } ;


    template<class TestObject>
    class KPCondStorage : public CondStorage<TestObject> {
      // Actual fields
      const KPeriodic word;
    public:
      // Implicit copy constructor on both
      KPeriodic getWord() const { return word ; }
      
    private:
      KPCondStorage(CondStorage<TestObject>&src) ;
      KPCondStorage() ;
    public:
      KPCondStorage(KPeriodic kpWord)
	: CondStorage<TestObject>(CondKPType),
	  word(kpWord) {} 
      virtual bool operator==(const CondStorage<TestObject>&c) const {
	if(c.getType() != CondStorage<TestObject>::getType()) return false ;
	const KPCondStorage<TestObject>& kpc = static_cast<const KPCondStorage<TestObject>&>(c) ;
	return (word == kpc.word) ;
      }
      virtual bool isComplement(const CondStorage<TestObject>&c) const {
	if(c.getType() != CondStorage<TestObject>::getType()) return false ;
	const KPCondStorage<TestObject>& kpc = static_cast<const KPCondStorage<TestObject>&>(c) ;
	return word.isComplement(kpc.word) ;
      }
      virtual CondStorage<TestObject>* buildComplement() const {
	KPCondStorage<TestObject>* obj =
	  new KPCondStorage<TestObject>(word.buildComplement()) ;
	CondStorage<TestObject>* res = static_cast<CondStorage<TestObject>*>(obj) ;
	return res ;
      }
      // Printing
      virtual void print(raw_ostream&os) const {
	os << word << " " ;
      }
      virtual void debugPrint(raw_ostream&os) {
	print(os) ;
      }
    } ;



    
    
    // I will always manipulate objects of this type,
    // which hide references inside. These references are
    // always allocated with "new".
    template<class TestObject>
    class Cond {
      // This pointer must always be valid
      CondStorage<TestObject>* impl ;
      // The empty and tombstone keys
      static EmptyCondStorage<TestObject> emptyKey ;
      static TombstoneCondStorage<TestObject> tombstoneKey ;

    private:
      Cond() ;
    public:
      // Copy constructor
      Cond(const Cond<TestObject>&src) : impl(src.impl) {} ;
      // Used internally to create Empty and Tombstone
      Cond(CondStorage<TestObject>*implIn) : impl(implIn) {}
      // Cond(EmptyCondStorage<TestObject>*implIn) : impl(implIn) {}
      Cond(KPeriodic word) {
	KPCondStorage<TestObject>* store =
	  new KPCondStorage<TestObject>(word) ;
	impl = static_cast<CondStorage<TestObject>*>(store) ;
      }
      Cond(TestObject value,bool whenotFlag) {
	DataCondStorage<TestObject>* store =
	  new DataCondStorage<TestObject>(whenotFlag,value) ;
	impl = static_cast<CondStorage<TestObject>*>(store) ;
      }

    public:
      //
      CondType getType() const { return impl->getType() ; }
      bool isNormal() const { return impl->isNormal() ; }
      bool getWhenotFlag() const {
	assert(getType() == CondDataType) ;
	const DataCondStorage<TestObject>* dc = static_cast<const DataCondStorage<TestObject>*>(impl);
	return dc->getWhenotFlag() ;
      }
      TestObject getData() const {
	assert(getType() == CondDataType) ;
	const DataCondStorage<TestObject>* dc = static_cast<const DataCondStorage<TestObject>*>(impl);
	return dc->getValue();
      }
      KPeriodic getWord() const {
	assert(getType() == CondKPType) ;
	const KPCondStorage<TestObject>* dc = static_cast<const KPCondStorage<TestObject>*>(impl);
	return dc->getWord() ;
      }
	
      
    public:
      // Hashing
      static const Cond<TestObject> getEmptyKey() ;
      static const Cond<TestObject> getTombstoneKey() ;
      unsigned getHashValue() const {
	if(impl->isNormal()) {
	  if(impl->getType()==CondDataType) {
	    const DataCondStorage<TestObject>* dc = static_cast<const DataCondStorage<TestObject>*>(impl) ;
	    return llvm::DenseMapInfo<TestObject>::getHashValue(dc->getValue())+0xabcd ;
	  } else {
	    const KPCondStorage<TestObject>* kpc = static_cast<const KPCondStorage<TestObject>*>(impl) ;
	    const KPeriodic& word = kpc->getWord() ;
	    const std::vector<bool>& prefix = word.getPrefix() ;
	    const std::vector<bool>& period = word.getPeriod() ;
	    unsigned result = 0xdef ;
	    for(int i=0;i<prefix.size();i++) {
	      result += prefix[i]?(1<<i):0 ;
	    }
	    for(int i=0;i<period.size();i++) {
	      result += period[i]?(1<<i):0 ;
	    }
	    return result ;
	  }
	} else return 0 ;
      }

      bool operator==(const Cond<TestObject>&c) const {
	return (*impl) == (*c.impl) ;
      }
      bool operator!=(const Cond<TestObject>&c) const {
	return !operator==(c) ;
      } 
      bool isComplement(const Cond<TestObject>&c)const {
	return impl->isComplement(*c.impl) ;
      }
      Cond<TestObject>* buildComplement() const {
	CondStorage<TestObject>* res_impl = impl->buildComplement() ;
	Cond<TestObject>* res = new Cond<TestObject>(res_impl) ;
	return res ;
      }
      void print(raw_ostream&os) const {
	impl->print(os) ;
      }
      void debugPrint(raw_ostream&os) const {
	impl->debugPrint(os) ;
      }
      
    } ;
  }
}

namespace llvm {
  // These DenseMapInfo objects are a sort of llvm-specific.
  // They must identify two objects of the given type to be used
  // for book keeping purposes (the EmptyKey and the TombstoneKey).
  // These are effectively out of normal use.
  // For this reason, one has to take care when using this data
  // structure with a scalar template argument.
  // For instance, DenseMapInfo<unsigned> uses the values
  // MAXINT and MAXINT-1 as EmptyKey and TombstoneKey (meaning that
  // I can use it to record Booleans).
  // For other types, I have to provide these, as well as a hash
  // function. For pair and tuple types, there are already constructors
  // provided.
  // The meaningful base class is found in include/llvm/ADT/DenseMapInfo.h
  
  // Type hash just like pointers.
  template <> struct DenseMapInfo<::mlir::lus::Cond<::mlir::Type>> {
    static ::mlir::lus::Cond<::mlir::Type> getEmptyKey() {
      return ::mlir::lus::Cond<::mlir::Type>::getEmptyKey() ;
    }
    static ::mlir::lus::Cond<::mlir::Type> getTombstoneKey() {
      return ::mlir::lus::Cond<::mlir::Type>::getTombstoneKey() ;
    }
    
    static unsigned getHashValue(const ::mlir::lus::Cond<::mlir::Type>& val) {
      return val.getHashValue() ;
    }
    
    static bool isEqual(const ::mlir::lus::Cond<::mlir::Type>& LHS,
			const ::mlir::lus::Cond<::mlir::Type>& RHS) {
      return LHS == RHS;
    }
  } ;

};

#endif
