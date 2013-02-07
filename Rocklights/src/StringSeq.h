#ifndef __STRINGSEQ_H__
#define __STRINGSEQ_H__

#include <string>
#include <vector>
#include <hash_map>
#include "SoundHash.h"
#include "globals.h"
using namespace std;
  

 /*
size_t std::hash_compare<std::string>::operator ()(const std::string& s) const {
	size_t h = 0;
    std::string::const_iterator p, p_end;
    for(p = s.begin(), p_end = s.end(); p != p_end; ++p)
    {
      h = 31 * h + (*p);
    }
    return h;
};
*/
  
class StringSeq {
public:
	bool isSeq;
	unsigned int curIndex;
	vector<string> ls;

	string name;

	typedef pair<string, unsigned int> entry;
	static hash_map<string,unsigned int> *indecies;

public:
	StringSeq(string propName, string emptyDef);
	~StringSeq();
	string getEl(); // get next or random depending on type
	string getNextEl();
	string getRandomEl();

	static void destroy() { if(indecies != NULL) delete  indecies; } // clears memeory

};

#endif
