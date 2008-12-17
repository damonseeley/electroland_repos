package net.electroland.connection.core;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

import net.electroland.udpUtils.UDPReceiver;

public class LightMonitor extends Thread{
	
	UDPReceiver receiver;
	boolean isRunning = true;
	
	public LightMonitor(int port) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
	}
	
	public void parseMsg(String msg) {
		System.out.println("msg: " + msg);
		// this really needs to be implemented to see what will come around
		// this may require doing a lower level implementation of a datagram socket
	}
	
	public void stopRunning() {
		isRunning = false;
		receiver.stopRunning();
	}

	public void run() {
		receiver.start();
		while (isRunning) {
			try {
				String msg = receiver.msgQueue.poll(2000, TimeUnit.MILLISECONDS);
				if (msg != null) { // make sure didn't time out
					parseMsg(msg);
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
}
