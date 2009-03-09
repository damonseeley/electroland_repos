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

	public UDPParser(int port) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
	}

	public void parseMsg(String msg) {
		//logger.debug("msg: " + msg);
		logger.info(msg);
		//String[] line = msg.split(",");
		// TODO check command byte against TCUtils command bytes
		//if((byte)Integer.parseInt(line[1]) == 0x00){
			
		//}
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
	public static void main(String[] args) throws SocketException, UnknownHostException {
		UDPParser parser = new UDPParser(10011) ;
		parser.start();
	}

}
