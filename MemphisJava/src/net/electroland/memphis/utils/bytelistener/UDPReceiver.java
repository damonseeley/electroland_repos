package net.electroland.memphis.utils.bytelistener;
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
	
	// logger
	//static Logger logger = Logger.getLogger(UDPReceiver.class);
	
	boolean isRunning = true;
	int highestPacketLength = 0;

	public LinkedBlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>();

	private  DatagramSocket receiveSocket;

	private DatagramPacket receivePacket;

	int receiveOnPort;

	public UDPReceiver(int receiveOnPort) throws SocketException,
			UnknownHostException {
		// this will cause malformed traxess errors when higher than 1024 character message!!!
		//receivePacket = new DatagramPacket(new byte[1024], 1024);
		receivePacket = new DatagramPacket(new byte[4096], 4096);
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
				//logger.info("packet length: "+receivePacket.getLength());
				if(receivePacket.getLength() > highestPacketLength){
					highestPacketLength = receivePacket.getLength();
					//logger.info("new largest packet: "+highestPacketLength);
				}
				//msgQueue.offer(new String(receivePacket.getData(), 0, receivePacket.getLength()));	// string
				msgQueue.offer(HexUtils.bytesToHex(receivePacket.getData(), receivePacket.getLength()));	// bytes
				if (reportedErr) {
					//logger.info("Moderator back up");
					reportedErr = false;
				}
			} catch (SocketTimeoutException e) {
				if (!reportedErr) {
					if (isRunning) {
						//logger.info("No messages from the moderator received in over 2 secs.");
						reportedErr = true;
					}
				}
			} catch (IOException e) {
				//logger.error(e.getMessage(), e);
			}
		}
		receiveSocket.close();
	}

	public void stopRunning() {
		isRunning = false;
	}
}
