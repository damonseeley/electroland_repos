package net.electroland.presenceGrid.net;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

public class GridDetectorClient extends Thread {
	
	boolean isRunning = true;
	GridDetectorDataListener listener;
	
	DatagramSocket datagramSocket;
	DatagramPacket receivePacket;
	byte[] data;
	
	public GridDetectorClient(int port, int bytes, GridDetectorDataListener listener) throws SocketException {
		this.listener = listener;
		data = new byte[bytes];
		receivePacket = new DatagramPacket(data, data.length);
		datagramSocket = new DatagramSocket(port);
	}
	
	public void run() {
		while(isRunning) {
			try {
				datagramSocket.receive(receivePacket);
				listener.receivedData(data);
				/*
				for(byte b : data) {
					System.out.print(b + " ");
				}
				System.out.println("");
				*/
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
	}
	
	
}
