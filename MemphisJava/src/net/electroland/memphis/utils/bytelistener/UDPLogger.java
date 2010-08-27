package net.electroland.memphis.utils.bytelistener;

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
	//static Logger logger = Logger.getLogger(UDPLogger.class);	
	
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
		//logger.debug("UDP logger running");
		System.out.println("UDP logger running");
		while (isRunning) {
			try {
				String msg = receiver.msgQueue.poll(2000, TimeUnit.MILLISECONDS);
				if (msg != null) { // make sure didn't time out
					logMsg(System.currentTimeMillis() + ": " + msg);
				}
			} catch (InterruptedException e) {
				System.out.println(e.getMessage() + " " + e);
				//logger.error(e.getMessage(), e);
			}
		}
		
	}


	public void logMsg(String msg) {
		if(isRunning) {
			try {
				System.out.println(msg);
				writer.write(msg);
				writer.newLine();
			} catch (IOException e) {
				System.out.println("Unable to write message");
				//logger.error("Unable to write message\n", e);
			}
		}
		
	}



	public static void main(String args[]) throws IOException, InterruptedException {
		// example usage
		UDPLogger logger = new  UDPLogger("fooLog.txt", 7474);
		logger.start();
		Thread.sleep(10 * (60 *1000)); // log messages for 10 minutes
		logger.stopRunning();
	}
}
