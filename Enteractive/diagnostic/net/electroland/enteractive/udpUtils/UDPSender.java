package net.electroland.enteractive.udpUtils;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

public class UDPSender {
	protected DatagramSocket socket;
	protected DatagramPacket packet;
	
	public UDPSender(String address, int port) throws UnknownHostException, SocketException{
		socket = new DatagramSocket();
		packet = new DatagramPacket(new byte[256], 256, InetAddress.getByName(address),port) ;	
	}
	
	public void sendString (String str) throws IOException {
		packet.setData(str.getBytes());
		socket.send(packet);
    }

	public void close() {
        socket.close();
	}

	
}
