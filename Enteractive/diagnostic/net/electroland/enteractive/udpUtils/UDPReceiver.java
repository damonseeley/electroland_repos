package net.electroland.enteractive.udpUtils;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;

import javax.swing.JTextArea;

import net.electroland.enteractive.diagnostic.LTOutputPanel;
import net.electroland.enteractive.utils.HexUtils;

/**
 * 	UDPReceiver just receives packets and sticks them on a msgQueue.  
 *  Be sure to call stopRunning() when done - this will close the socket properly.
 *
 * @author eitan
 *
 */
public class UDPReceiver extends Thread {
	boolean isRunning = true;

	public LinkedBlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>();

	private  DatagramSocket receiveSocket;

	private DatagramPacket receivePacket;

	int receiveOnPort;

	public UDPReceiver(int receiveOnPort) throws SocketException, UnknownHostException {
		receivePacket = new DatagramPacket(new byte[128], 64);
		this.receiveOnPort = receiveOnPort;
		receiveSocket = new DatagramSocket(receiveOnPort);
	}

	public void run() {
		//boolean reportedErr = false;
		try {
			// timeout and report error if not meesages from moderator in 2 secs
			receiveSocket.setSoTimeout(2000);
		} catch (SocketException e1) {
			// won't allow setting not really important
		}

		while (isRunning) {
			try {
				receiveSocket.receive(receivePacket);
				msgQueue.offer(new String(receivePacket.getData(), 0, receivePacket.getLength()));
				byte[] b = receivePacket.getData();
				//HexUtils.printHex(b);
				//feedOutput(b, receivePacket.getLength());
				
			} catch (SocketTimeoutException e) {
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		receiveSocket.close();
	}


	public void stopRunning() {
		isRunning = false;
	}
}
