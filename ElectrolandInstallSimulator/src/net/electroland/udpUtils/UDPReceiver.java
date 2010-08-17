package net.electroland.udpUtils;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;


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

	public UDPReceiver(int receiveOnPort) throws SocketException,
			UnknownHostException {
		receivePacket = new DatagramPacket(new byte[1024], 1024);
		this.receiveOnPort = receiveOnPort;
		receiveSocket = new DatagramSocket(receiveOnPort);
	}

	public void run() {
		boolean reportedErr = false;
		try {
			// timeout and report error if not meesages from moderator in 2 secs
			receiveSocket.setSoTimeout(2000);
		} catch (SocketException e1) {
			// won't allow setting not really important
		}

		while (isRunning) {
			try {
				receiveSocket.receive(receivePacket);
				msgQueue.offer(new String(receivePacket.getData(), 0,
						receivePacket.getLength()));
				if (reportedErr) {
					System.err.println("Moderator back up");
					reportedErr = false;
				}
			} catch (SocketTimeoutException e) {
				if (!reportedErr) {
					if (isRunning) {
						System.err
								.println("No messages from the moderator received in over 2 secs.");
						reportedErr = true;
					}
				}
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
