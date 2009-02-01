package net.electroland.enteractive.udpUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Be sure to call stopRunning or you may not get all the messages written to the file
 * @author eitan
 *
 */
public class UDPLogger extends Thread {
	BufferedWriter writer;
	
	UDPReceiver receiver;
	boolean isRunning = true;

	public UDPLogger(String filename, int port) throws IOException {
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
		while (isRunning) {
			try {
				String msg = receiver.msgQueue.poll(2000, TimeUnit.MILLISECONDS);
				if (msg != null) { // make sure didn't time out
					logMsg(msg);
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
	}


	public void logMsg(String msg) {
		if(isRunning) {
			try {
				//System.out.println(msg);
				writer.write(msg);
				writer.newLine();
			} catch (IOException e) {
				System.err.println("Unable to write message\n" + e);
			}
		}
		
	}



	public static void main(String args[]) throws IOException, InterruptedException {
		// example usage
		UDPLogger logger = new  UDPLogger("fooLog.txt", 10001);
		logger.start();
		//Thread.sleep(10 * (60 *1000)); // log messages for 10 minutes
		Thread.sleep(1 * (10 *1000)); // log messages for 10 minutes
		System.out.println("Stopping");
		logger.stopRunning();
	}
}
