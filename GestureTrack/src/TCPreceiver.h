/**************************************************************************
 *
 *	TCPreceiver.h
 *
 *	Description	:	Class definitions for demo code for TCP sending of images
 *
 *	Copyright (c) 2008-2009, Tyzx Corporation. All rights reserved.
 *
 **************************************************************************/
# ifndef TCP_RECEIVER_H_
# define TCP_RECEIVER_H_

class TCPreceiver
{
public:
	TCPreceiver();
	~TCPreceiver();
	int connect(const char *servIP, int port);
	int closeConnection();
	int send(void *buffer, int lengthInBytes);
	int recv(void *buffer, int expectedLengthInBytes);
	void getBytesTransmitted(unsigned long &sent, unsigned long &received);
protected:
	char ipAddress[50];
	int error(const char *message);
	int servSock;     
	unsigned long bytesReceived;
	unsigned long bytesSent;
};
# endif
