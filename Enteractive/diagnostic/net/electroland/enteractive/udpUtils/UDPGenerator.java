package net.electroland.enteractive.udpUtils;

// test in FEB 08!!!

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

/**
 * Reads logfile (created by UDPLogger) and sends out upd messages to specified port
 * @author eitan
 *
 */

public class UDPGenerator extends Thread {
	String filename;
	protected boolean isRunning = true;
	protected UDPSender updSender;
	protected long frameLength;
	
	protected int loopCnt = 1;
	
	public UDPGenerator(String filename, String address, int port, long frameLength) throws UnknownHostException, SocketException  {
		updSender = new UDPSender(address, port);
		this.frameLength = frameLength;
		this.filename = filename;
	}
	
	public void stopRunning() {
		isRunning = false;
	}
	
	public boolean isRunning() {
		return isRunning;
	}
	
	
	/**
	 * @param loopCnt - numer of times to read the file.  Use -1 to loop forever
	 */
	public void setLoopCount(int loopCnt) {
		this.loopCnt = loopCnt;
	}
	public void run() {
		try {
			long sleepTime = 0;
			long curTime = 0;
			long lastMsg = 0;
			while(isRunning && (loopCnt-- != 0)) {
				BufferedReader reader = new BufferedReader(new FileReader(filename));
				while(isRunning && reader.ready()) {
					lastMsg = System.currentTimeMillis();
					updSender.sendString(reader.readLine());
					curTime = System.currentTimeMillis();
					sleepTime = frameLength - (curTime - lastMsg);
					if(sleepTime >= 0) {
						try {
							synchronized(this) {
								wait(sleepTime);
							}
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
					}
					
				}
				reader.close();				
			}
			isRunning = false;
			updSender.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws IOException, InterruptedException {
		// test
		UDPGenerator generator = new  UDPGenerator("fooLog.txt", "localhost",  5432, 35);
		generator.setLoopCount(3);
		UDPReceiver rec = new UDPReceiver(5432);
		rec.start();
		generator.start();
		
		while(generator.isRunning()) { // wait until its done
			Thread.sleep(1000);
		}
		
		rec.stopRunning();
		generator.stopRunning();
	}


}
