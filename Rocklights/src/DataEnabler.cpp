/*
 *  DataEnabler.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#include "DataEnabler.h"




DataEnabler::DataEnabler() {
  size=0;
  address=NULL;
}

DataEnabler::DataEnabler(char* ip, int dataSize) {
  size = dataSize;
  address = ip;
  data = new unsigned char[size];
  if (data == NULL) {
	timeStamp(); clog << "ERROR  Unable to create DataEnabler with ip " << ip << 
						 " of size " << dataSize << std::endl;
			Globals::hasError = true;

  }
  clear();
}

void DataEnabler::set(char* ip, int dataSize) {
  size = dataSize;
  address = ip;
  data = new unsigned char[size];
  if (data == NULL) {
    timeStamp(); clog << "ERROR  Unable to create DataEnabler with ip " << ip << " of size " << dataSize << std::endl;
			Globals::hasError = true;

  }
  clear();
}
DataEnabler::~DataEnabler() {
  delete address;
  delete[] data;
}

void DataEnabler::clear() {
  memset(data, 0, sizeof(unsigned char) * size);
}

/*
void DataEnabler::print() {
  for(int i = 0; i < size - 1; i++) {
    std::cou t << (int) data[i] << ", ";
  }
  std::cou t << (int) data[size-1] << " -> " << address << std::cout;
}
void DataEnabler::printNonZero() {
  for(int i = 0; i < size; i++) {
    if (data[i] != 0) {
      std::cou t << "_" << address << ":" << i << "<<" << (int) data[i] << endl;
    }
  }

}
*/
/*
void DataEnabler::println() {
  print();
  std::cou t << std::endl;
}
*/
void DataEnabler::sendDMX() {
  if(address == NULL) return;
//    printNonZero();
  DataEnablers::ks->sendDMXFrame(address, 0, 0, data, size);
}
