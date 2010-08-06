#include "UDPSender.h"
#include <iostream>
#include <sstream>
#include <stdio.h>

UDPSender::UDPSender(string ipStr, int port) {

		WORD wVersionRequested = MAKEWORD(1,1);
	WSADATA	wsaData;
	int	nRet;
	//
	// Initialize WinSock and check	the	version
	//
	nRet = WSAStartup(wVersionRequested, &wsaData);


	if (wsaData.wVersion !=	wVersionRequested)
	{	
		std::cout << "Wrong version or winsock";
	}

		char* ip = new char[ipStr.size() + 1];
	strcpy(ip, ipStr.c_str());


	printf("\nTUB sending to: %s on	port: %d", ip, port);
	LPHOSTENT lpHostEntry;

	lpHostEntry	= gethostbyname(ip);
	if (lpHostEntry	== NULL)
	{
		std::cout <<"ip address is invalid";
	}

	// Create a	UDP/IP datagram	socket
	//

	theSocket =	socket(AF_INET,			// Address family
		SOCK_DGRAM,		// Socket type
		IPPROTO_UDP);	// Protocol


	if (theSocket == INVALID_SOCKET)
	{
		theSocket =	NULL;
		std::cout <<"unable to open socket";
	}


	//
	// Fill	in the address structure for the client
	//

	saServer.sin_family	= AF_INET;
	saServer.sin_addr =	*((LPIN_ADDR)*lpHostEntry->h_addr_list);
	// ^ Server's address
	saServer.sin_port =	htons(port);	// Port	number from	command	line


}


void UDPSender::sendString(const char *szBuf)
{
	int	nRet;

	nRet = sendto(theSocket,				// Socket
		szBuf,					// Data	buffer
		strlen(szBuf),			// Length of data
		0,						// Flags
		(LPSOCKADDR)&saServer,	// Server address
		sizeof(struct	sockaddr));	// Length of address
	if (nRet ==	SOCKET_ERROR)
	{
		std::cerr <<"socket error sending UPD.  Closing socket";
		closesocket(theSocket);
		theSocket =	NULL;
		return;
	}
}

