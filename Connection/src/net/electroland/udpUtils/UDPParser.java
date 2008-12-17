package net.electroland.udpUtils;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;


public class UDPParser extends Thread {

	// logger
	static Logger logger = Logger.getLogger(UDPParser.class);	
	
	UDPReceiver receiver;
	boolean isRunning = true;

	public void handleEnter(int id) {
		//logger.debug("handling Enter " + id);
	}

	public void handleTrackInfo(int id, int x, int y, int h) {
		//logger.debug("handling TrackData " + id + "(" + x +", " + y + ", " + h + ")");
	}

	public void handleExit(int id) {
		//logger.debug("handling Exit " + id);
	}

	public UDPParser(int port) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
	}

	public void parseMsg(String msg) {
		//logger.debug("msg: " + msg);
		String rest = msg;
		try {
			while (rest.charAt(0) != ';') {
				String[] result = rest.split(",", 2);
				handleEnter(Integer.parseInt(result[0]));
				rest = result[1];
			}
			rest = rest.substring(1);
			while (rest.charAt(0) != ';') {
				String[] result = rest.split(",", 5);
				handleTrackInfo(Integer.parseInt(result[0]), Integer
						.parseInt(result[1]), Integer.parseInt(result[2]),
						Integer.parseInt(result[3]));
				rest = result[4];
			}
			rest = rest.substring(1);
			while ((rest != null) && (rest.length() > 0)) {
				String[] result = rest.split(",", 2);
				handleExit(Integer.parseInt(result[0]));
				if (result.length >= 2) {
					rest = result[1];
				} else {
					rest = null;
				}
			}
		} catch (RuntimeException e) {
			logger.error("Malformed Traxess data.\n   Parsing error caught near :"
							+ rest + "\n   in msg:" + msg + "\n", e);
		}
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
				logger.error(e.getMessage(), e);
			}
		}
	}




	//just for testing
	//	public static void main(String[] args) throws SocketException, UnknownHostException {
	//		UDPParser parser = new UDPParser(4114) ;
	//		parser.start();
	//	}

}
