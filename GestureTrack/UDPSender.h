#ifndef __UDPSENDER_H__
#define __UDPSENDER_H__

#include <winsock.h>
#include <string>

#define DEFAULT_PORT 4114
#define DEFAULT_IP "localhost"


using namespace std;

class UDPSender {
public:
	int  buffSize;

	SOCKET	theSocket;
	SOCKADDR_IN	saServer;

	UDPSender(string ip, int port);
	~UDPSender();

	void sendString(const char *szBuf);



}
;

#endif