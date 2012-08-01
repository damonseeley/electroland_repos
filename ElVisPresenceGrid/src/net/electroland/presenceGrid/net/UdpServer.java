package net.electroland.presenceGrid.net;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

public class UdpServer {
	
	DatagramSocket datagramSocket;
	DatagramPacket sendPacket;
	byte[] data;
	
	public UdpServer(String clientIp, int port, byte[] data) throws SocketException, UnknownHostException {
		
		InetAddress host = InetAddress.getByName(clientIp);
		this.data = data;
		sendPacket = new DatagramPacket(data, data.length, host, port) ;

		datagramSocket = new DatagramSocket();

	}
	
	public void send() throws IOException {
		datagramSocket.send(sendPacket);
	}
	



}
