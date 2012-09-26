package net.electroland.elvis.net;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;


public class UDPBroadcaster extends Thread {
	LinkedBlockingQueue<StringAppender> objQueue;
	DatagramSocket socket;
	DatagramPacket packet;
	public boolean isRunning = true;
	StringBuilder buffer;
	byte[] bytes = new byte[65507];

	public UDPBroadcaster(int port) throws SocketException, UnknownHostException {
		this("localhost", port);
	}

	public UDPBroadcaster(String address, int port) throws SocketException, UnknownHostException {
		super();
		//System.out.println("UDPBroadcaster address: " + address + " port: " + port);
		buffer = new StringBuilder();
		socket = new DatagramSocket();
		if(address.equals("broadcast")) {
			socket.setBroadcast(true);
			packet = new DatagramPacket(bytes, 0, InetAddress.getByName("255.255.255.255"), port);
		} else {
			packet = new DatagramPacket(bytes, 0, InetAddress.getByName(address), port);			
		}
		objQueue = new LinkedBlockingQueue<StringAppender> ();
	}
	
	
	public void send(StringAppender objs) {
		objQueue.offer(objs);
	}

	public void stopRunning() {
		isRunning = false;
		objQueue.add(new StringAppender.EmptyStringAppender());
	}
	
	public void run() {
		while (isRunning) {
			try {
				StringAppender vec = objQueue.take();
				broadcast(vec);				
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}	
	}

	public void broadcast(StringAppender sa) {
		buffer.setLength(0); //reset
		sa.buildString(buffer);
		packet.setData(buffer.toString().getBytes());
		try {
			//System.out.println("Sending UDP on port: " + socket.getPort());
			socket.send(packet);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
