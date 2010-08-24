package net.electroland.memphis.utils.bytelistener;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

public class UDPParser extends Thread {

	// logger
	//static Logger logger = Logger.getLogger(UDPParser.class);	
	
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
		System.out.println(msg);
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
				//logger.error(e.getMessage(), e);
			}
		}
	}

}
