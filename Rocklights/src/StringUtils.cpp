#include "StringUtils.h"
#include "globals.h"

int StringUtils::split(const string& input, vector<string>& result) {
	int pos = 0;
	int commaPos = input.find(",", pos);
	if(commaPos == string::npos) { // no comma
		result.push_back(input);
		return 0;
	} 
	string s;
	while(commaPos != string::npos) {
		s = input.substr( pos, commaPos-pos);
		result.push_back(s);
		pos = commaPos +1;
		commaPos = input.find(",", pos);
		// strip leading spaces
		while((input[pos] == ' ') || (input[pos] == ',')) {
			pos++;
		}
	}
	s = input.substr(pos);
	result.push_back(s);
	return result.size();
}

string StringUtils::pickRandom(vector<string>& input) {
	return input[random(input.size())];
}

int* StringUtils::createIntArray(vector<string> vec) {
	int *ar= new int[vec.size()];
	for(unsigned int i = 0; i < vec.size(); i++) {
		ar[i] = atoi(vec[i].c_str());
	}
	return ar;
}

float* StringUtils::createFltArray(vector<string> vec) {
	float *ar= new float[vec.size()];
	for(unsigned int i = 0; i < vec.size(); i++) {
		ar[i] = (float) atof(vec[i].c_str());
	}
	return ar;
}