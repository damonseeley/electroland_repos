package net.electroland.elvis.net;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.util.concurrent.LinkedBlockingQueue;

public abstract class UDPClient<T> extends Thread {
	DatagramSocket socket;
	DatagramPacket packet;
	boolean isRunning = true;
	byte[] bytes = new byte[65508];
	StringBuffer sb;
	LinkedBlockingQueue<String> receivedStringQueue = new LinkedBlockingQueue<String>();
	LinkedBlockingQueue<T> receivedObject = new LinkedBlockingQueue<T>();
	
	public UDPClient(int port) throws SocketException {
		super();
		socket = new DatagramSocket(port);
		packet = new DatagramPacket(bytes, bytes.length);
		new ParserThread().start();
		new ObjHandlerThread().start();
	}
	
	public void stopRunning() {
		isRunning = false;
		receivedStringQueue.offer("");
	}
	
	public void run() {
		while(isRunning) {
			try {
				socket.receive(packet);
				String s = null;
				if(packet.getLength() == 0) {
					s = "";
				} else {
					s = new String(packet.getData());
				}
				receivedStringQueue.offer(s);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
	}
	public abstract T parse(String s) ;
	
	public abstract void handle(T t);
	
	
	public class ObjHandlerThread extends Thread {
		public ObjHandlerThread() {
			super();
		}
		public void run() {
			while(isRunning) {
				try {
					handle(receivedObject.take());
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}
	public class ParserThread extends Thread {
		public ParserThread() {
			super();
		}
		public void run() {
			while(isRunning) {
				try {
					String s = receivedStringQueue.take();
					receivedObject.offer(parse(s));
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	

}
