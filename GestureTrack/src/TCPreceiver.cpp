/**************************************************************************
 *
 *	TCPreceiver.cpp
 *
 *	Description	:	Demo code for TCP sending of images
 *
 *	Copyright (c) 2008-2009, Tyzx Corporation. All rights reserved.
 *
 **************************************************************************/

# include <stdio.h>
# include <winsock.h>    /* for socket(),... */
# include <stdlib.h>     /* for exit() */

# include "TCPreceiver.h"

TCPreceiver::TCPreceiver(): servSock(-1), bytesSent(0), bytesReceived(0)
{
}

TCPreceiver::~TCPreceiver()
{
	closeConnection();
}

int 
TCPreceiver::connect(const char *servIP, int port)
{
	struct sockaddr_in servAddr; /* server address */
# ifdef WIN32
	WSADATA wsaData;                 

	if (WSAStartup(MAKEWORD(2, 0), &wsaData) != 0) 
	{
		fprintf(stderr, "WSAStartup() failed.");
		exit(1);
	}
# endif
 
	/* Create socket for talking to server */
	if ((servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
		return error("socket() failed.");

	int flag = 1;

	if (setsockopt(servSock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag)) < 0)
		return error("TCP_NODELAY failed.");

	struct hostent	*h;

	h = gethostbyname(servIP);
	if (!h) {
		return error("gethostbyname failed.");
	}

	/* fill in address, connect to port, 
	 * send size declaration
	 */		
	
    memset(&servAddr, 0, sizeof(servAddr));     /* Zero out structure */
    servAddr.sin_family      = AF_INET;             /* Internet address family */
    servAddr.sin_addr.s_addr = *(unsigned long *)h->h_addr;   /* Server IP address */
    servAddr.sin_port        = htons(port); /* Server port */

//	bind
    /* Establish the connection to the echo server */
	int errmsg = ::connect(servSock, (struct sockaddr *) &servAddr, sizeof(servAddr));
	if (errmsg < 0) {
		errmsg = WSAGetLastError();
        return error("connect() failed.");
	}

	return 0;
}

int 
TCPreceiver::closeConnection()
{
# ifdef WIN32
	if (servSock >= 0) 
		closesocket(servSock);
	WSACleanup();  /* Cleanup Winsock */
# else
	if (servSock >= 0) 
		close(servSock);
# endif
	servSock = -1;

	return 0;
}

int 
TCPreceiver::send(void *buffer, int lengthInBytes)
{
	if (::send(servSock, (char *) buffer, lengthInBytes, 0) != lengthInBytes)
		return error("send() failed.");

	bytesSent += lengthInBytes;

	return 0;
}

int 
TCPreceiver::recv(void *buffer, int expectedLengthInBytes)
{
	int remainingLength = expectedLengthInBytes;
	char *insertion = (char *) buffer;
	do {
		int len = ::recv(servSock, insertion, remainingLength, 0);

		if (len <= 0) {
			// zero means close, negative is error
			return error((len == 0) ? "Connection closed." : "recv() failed.");
		}
		bytesReceived += len;
		insertion += len;
		remainingLength -= len;
	} while (remainingLength > 0);

	return 0;
}

void 
TCPreceiver::getBytesTransmitted(unsigned long &sent, unsigned long &received)
{
	sent = bytesSent;
	received = bytesReceived;
}

int 
TCPreceiver::error(const char *message)
{
	fprintf(stderr, "%s\n", message);
	return -1;
}
