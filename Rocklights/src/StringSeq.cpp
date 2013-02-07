#include "StringSeq.h"
#include "Profile.h"
#include "StringUtils.h"
#include "globals.h"

hash_map<string,unsigned int> *StringSeq::indecies = NULL;

StringSeq::StringSeq(string propName, string emptyDef) {
	if (indecies == NULL) {
		indecies = new hash_map<string, unsigned int>();
	}
	name = propName;
	hash_map<string,unsigned int>::iterator it;
	it = indecies->find(propName);
	if(it != indecies->end()) {

		curIndex = it->second;
	} else {
		curIndex = -1;
	}
  

	StringUtils::split(CProfile::theProfile->String(propName.c_str(), emptyDef.c_str()), ls);
	if(ls[ls.size() -1] == "_S_") {
		isSeq = true;
		ls.pop_back();
	} else {
		isSeq = false;
	}
}

StringSeq::~StringSeq() {
	ls.clear();
	hash_map<string,unsigned int>::iterator it;
	it = indecies->find(name);
	if(it != indecies->end()) {

		it->second = curIndex;
//	cout << "   reading " << name << " " << curIndex << endl;
	} else {
	indecies->insert(entry(name,curIndex));
	}
}

string StringSeq::getNextEl() {
	curIndex++;
	curIndex = (curIndex >= ls.size()) ? 0 : curIndex;
	return ls[curIndex];
}

string StringSeq::getRandomEl() {
	return ls[random(ls.size())];
}
string StringSeq::getEl() {
	if(isSeq) {
		return getNextEl();
	} else {
		return getRandomEl();
	}
}
