#ifndef __ERROR_LOG__
#define __ERROR_LOG__

#include <string>
#include <iostream>
#include <fstream>

#define DEFAULT_ERROR_LOG "ELPTrackError.log"

class ErrorLog {

public:
	bool useStdErr;
	bool useStream;

	std::ofstream stream;

	static ErrorLog *log;

	static void setLogFile(std::string logName);
	static void close();


	void streamDataTime();


	template<typename T>
	ErrorLog& operator << (const T& object)
	{

		if(useStream) 
			stream << object;

		if(useStdErr)
			std::cerr << object;

		return *this;
	};

	typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

	typedef CoutType& (*StandardEndLine)(CoutType&);

	// define an operator<< to take in std::endl

	ErrorLog& operator<<(StandardEndLine manip)
	{
		streamDataTime();
		if(useStream)
			manip(stream);

		if(useStdErr)
			manip(std::cerr);
		return *this;
	}

private:
	ErrorLog();
	ErrorLog(std::string logName, bool useStnOut = true);
	~ErrorLog();
};


#endif
