package net.electroland.udpUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;

/**
 * Be sure to call stopRunning or you may not get all the messages written to the file
 * @author eitan
 *
 */
public class UDPLogger extends Thread {

	// logger
	static Logger logger = Logger.getLogger(UDPLogger.class);	
	
	BufferedWriter writer;
	
	UDPReceiver receiver;
	boolean isRunning = true;

	public UDPLogger(String filename, int port) throws IOException {
		// should convert this to a Log4j RollingFileAppender.
		writer = new BufferedWriter(new FileWriter(filename, false));
		receiver = new UDPReceiver(port);
	}


	public void stopRunning() throws IOException {
		isRunning = false;
		receiver.stopRunning();
		writer.flush();
		writer.close();
	}

	public void run() {
		receiver.start();
		logger.debug("UDP logger running");
		while (isRunning) {
			try {
				String msg = receiver.msgQueue.poll(2000, TimeUnit.MILLISECONDS);
				if (msg != null) { // make sure didn't time out
					logMsg(msg);
				}
			} catch (InterruptedException e) {
				logger.error(e.getMessage(), e);
			}
		}
		
	}


	public void logMsg(String msg) {
		if(isRunning) {
			try {
				writer.write(msg.toString());	// don't think this will work right
				writer.newLine();
			} catch (IOException e) {
				logger.error("Unable to write message\n", e);
			}
		}
		
	}



	public static void main(String args[]) throws IOException, InterruptedException {
		// example usage
		UDPLogger logger = new  UDPLogger("fooLog.txt", 5432);
		logger.start();
		Thread.sleep(10 * (60 *1000)); // log messages for 10 minutes
		logger.stopRunning();
	}
}
