#include "ErrorLog.h"

#include <iomanip>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>


ErrorLog *ErrorLog::log = new ErrorLog();  // is there any way to init the var to null?

ErrorLog::ErrorLog() {	
	this->useStdErr = true;
	this->useStream = false;
}

ErrorLog::ErrorLog(std::string logName, bool useStdOut) {	
	this->useStdErr = useStdOut;
	this->useStream = true;
	stream.open(logName, std::ios_base::app);
}

ErrorLog::~ErrorLog() {	
	if(useStream) 
		stream.close();
}

void ErrorLog::setLogFile(std::string logName) {
	close();
	log = new ErrorLog(logName);
}


void ErrorLog::close() {
	if(log) {
		delete log;
		log = NULL;
	}
}

void ErrorLog::streamDataTime() {
	time_t rawtime;
	struct tm timeinfo;
	char buffer [80];

	time ( &rawtime );
	localtime_s (&timeinfo,&rawtime);
	strftime (buffer,80,"    [ %Y/%m/%d  %H::%M::%S ]",&timeinfo);
	if(useStream)
		stream << std::setw(60) << buffer;
	//std out doesn't need timestamps!
	//	if(useStdOut)
//		std::cout << std::setw(60) << buffer;
}


