#ifndef __SPLITSTRING_H__
#define __SPLITSTRING_H__

#include <string>
#include <vector>

using namespace std;

class StringUtils
{

public:

	static int split(const string& input, vector<string>& result);
	static string pickRandom(vector<string>& input);
	static int* createIntArray(vector<string> vec);
	static float* createFltArray(vector<string> vec);


};

#endif