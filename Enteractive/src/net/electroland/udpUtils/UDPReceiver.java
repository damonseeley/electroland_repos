package net.electroland.udpUtils;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;
import net.electroland.enteractive.utils.HexUtils;
import org.apache.log4j.Logger;


/**
 * 	UDPReceiver just receives packets and sticks them on a msgQueue.  
 *  Be sure to call stopRunning() when done - this will close the socket properly.
 *
 * @author eitan (modified by asiegel)
 */
public class UDPReceiver extends Thread {
	
	// logger
	static Logger logger = Logger.getLogger(UDPReceiver.class);
	
	boolean isRunning = true;
	int highestPacketLength = 0;

	public LinkedBlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>();

	private  DatagramSocket receiveSocket;

	private DatagramPacket receivePacket;

	int receiveOnPort;

	public UDPReceiver(int receiveOnPort) throws SocketException,
			UnknownHostException {
		receivePacket = new DatagramPacket(new byte[4096], 4096);
		this.receiveOnPort = receiveOnPort;
		receiveSocket = new DatagramSocket(receiveOnPort);
	}

	public void run() {
		while (isRunning) {
			try {
				receiveSocket.receive(receivePacket);
				//logger.info("packet length: "+receivePacket.getLength());
				//if(receivePacket.getLength() > highestPacketLength){
				//	highestPacketLength = receivePacket.getLength();
				//	logger.info("new largest packet: "+highestPacketLength);
				//}
				msgQueue.offer(HexUtils.bytesToHex(receivePacket.getData(), receivePacket.getLength()));
			} catch (IOException e) {
				logger.error(e.getMessage(), e);
			}
		}
		receiveSocket.close();
	}

	public void stopRunning() {
		isRunning = false;
	}
}
