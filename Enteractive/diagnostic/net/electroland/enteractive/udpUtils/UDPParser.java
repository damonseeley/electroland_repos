package net.electroland.enteractive.udpUtils;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;


public class UDPParser extends Thread {

	UDPReceiver receiver;
	boolean isRunning = true;

	public void handleEnter(int id) {
		//System.out.println("handling Enter " + id);
	}

	public void handleTrackInfo(int id, int x, int y, int h) {
		//System.out.println("handling TrackData " + id + "(" + x +", " + y + ", " + h + ")");
	}

	public void handleExit(int id) {
		//System.out.println("handling Exit " + id);
	}

	public UDPParser(int port) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
	}

	public void parseMsg(String msg) {
		//System.out.println("msg: " + msg);
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
			System.err.println("Malformed Traxess data.\n   Parsing error caught near :"
							+ rest + "\n   in msg:" + msg + "\n");
			e.printStackTrace();
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
				e.printStackTrace();
			}
		}
	}




	//just for testing
	//	public static void main(String[] args) throws SocketException, UnknownHostException {
	//		UDPParser parser = new UDPParser(4114) ;
	//		parser.start();
	//	}

}
