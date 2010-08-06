#ifndef __UDPRECEIVER_H__
#define __UDPRECEIVER_H__

#include <winsock.h>


#define DEFAULT_PORT 4114
#define DEFAULT_IP "localhost"



class UDPReceiver {
public:
	SOCKET	theSocket;
	SOCKADDR_IN	saServer;

	UDPReceiver( int port);
	~UDPReceiver();

	char* get();



}
;
#endif