#include "KPeriodic.h"

namespace mlir {
  namespace lus {

    const KPeriodic KPeriodic::initWord(std::vector<bool>({true}),
				  std::vector<bool>({false})) ;
    const KPeriodic KPeriodic::preWord(std::vector<bool>({false}),
				  std::vector<bool>({true})) ;
    
    KPeriodic* parseKPeriodic(llvm::StringRef str) {
      int lparenPosition, rparenPosition ;
      for(lparenPosition=0;
	  (lparenPosition<str.size())&&(str[lparenPosition]!='(');
	  lparenPosition++) ;
      assert(lparenPosition != str.size()) ;
      for(rparenPosition=0;
	  (rparenPosition<str.size())&&(str[rparenPosition]!=')');
	  rparenPosition++) ;
      assert(rparenPosition != str.size()) ;
      std::vector<bool> prefix;
      for(int i=0;i<lparenPosition;i++){
	if(str[i]=='0') prefix.push_back(false) ;
	else if(str[i]=='1') prefix.push_back(true) ;
	else assert(false) ;
      }
      std::vector<bool> period;
      for(int i=lparenPosition+1;i<rparenPosition;i++){
	if(str[i]=='0') period.push_back(false) ;
	else if(str[i]=='1') period.push_back(true) ;
	else assert(false) ;
      }
      // Make the k-periodic word permanent
      return
	new KPeriodic(*(new std::vector<bool>(prefix)),
		      *(new std::vector<bool>(period))) ;
    }
  }
}
