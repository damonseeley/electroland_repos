/*
 *  DataEnablerFile.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#include "DataEnablerFile.h"


DataEnablerFile::DataEnablerFile(const char *filename) {
  filenameStr = filename;

  
  if(open(filename)) {
	  timeStamp(); clog << "INFO  Opened file: " << filename << endl;
  } else {
      timeStamp(); clog << "ERROR  unable to open file: " << filename << endl;
	  		Globals::hasError = true;

  }
  maxid = -1;
  line = 0;
  
  for(int i = 0; i < MAXENABLERS; i++) {
    *ip[i] = '\0';
    size[i] = 0;
  }
  line = 0;
  
}

bool DataEnablerFile::open(const char *filename) {
  file.open(filename, ios::in);
  return file.is_open();
}

DataEnablerFile::~DataEnablerFile() {
  file.close();
}

void DataEnablerFile::reopen() {
  close();
  file.clear();
  open(filenameStr.c_str());
}  

bool DataEnablerFile::isWhitespace(const char& c)
{
     return (c == ' ' || c == '\t');     //add other whitespace chars as necessary
}

string DataEnablerFile::trim(string x)
{
     while(isWhitespace(x[0])) x.erase(0, 1);     //trim leading whitespace

     basic_string<char, std::char_traits<char> >::iterator i = x.begin();

     while(i != x.end())
     {
          if(isWhitespace(*i)) i = x.erase(i);
          else i++;
     }

     return x;
}
bool DataEnablerFile::readLine(int &id, string &ip, int &size) {
  string ipStr;
  string idStr;
  string sizeStr;
  

  if (getline(file, idStr, ',').eof()) return false;
  if (getline(file, ipStr, ',').eof()) return false;

  string::size_type  notwhite = ipStr.find_first_not_of(" "); // strip leading whitespace
  ipStr.erase(0,notwhite);
//  notwhite = ipStr.find_last_not_of(" ");
//  ipStr.erase(notwhite+1);

  if (getline(file, sizeStr).eof()) return false;
  
  id = atoi(idStr.c_str());
  ip = ipStr;
  size = atoi(sizeStr.c_str());
  line++;
  
  
  return true;
}

void DataEnablerFile::read() {
  int newId;
  string newIp;
  int newSize;
    while(readLine(newId, newIp, newSize)) {
      if(newId >= MAXENABLERS) {
        timeStamp(); clog << "ERROR  unable to read in data enablers.  MAXENABLERS set too low for ID " << newId << endl;
				Globals::hasError = true;

      }
      strcpy(ip[newId], newIp.c_str());
      size[newId] = newSize;
      maxid = (newId > maxid) ? newId : maxid;
    }
}
/*
void DataEnablerFile::print() {
    for(int i = 0; i <= maxid; i++) {
      cou t << i << ")  " << ip[i] << "   size:" << size[i] << endl;
    }
}
*/