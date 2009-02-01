package net.electroland.enteractive.diagnostic;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import net.electroland.enteractive.utils.HexUtils;

/**
 * Be sure to call stopRunning or you may not get all the messages written to the file
 * @author eitan
 *
 */
public class LTUDPLogger extends Thread {
	BufferedWriter writer;
	
	LTUDPReceiver receiver;
	int port;
	boolean isRunning = true;
	
	long lastTime = 0;
	

	public LTUDPLogger(String filename, int port) throws IOException {
		this.port = port;
		writer = new BufferedWriter(new FileWriter(filename, false));
		receiver = new LTUDPReceiver(this.port);
	}

	public void stopRunning() throws IOException {
		isRunning = false;
		receiver.stopRunning();
		writer.flush();
		writer.close();
	}

	public void run() {
		receiver.start();
		System.out.println("logger listening on port " + port);
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
				long thisTime = System.currentTimeMillis();
				long interval = thisTime - lastTime;
				writer.write(msg + "  interval: " + interval);
				lastTime =  thisTime;
				writer.newLine();
			} catch (IOException e) {
				System.err.println("Unable to write message\n" + e);
			}
		}
		
	}
	
	public void registerOutput(LTOutputPanel ltto) {
		//this.ltto = ltto;
		receiver.registerOutput(ltto);
		//outputTA = ltto.getOutputField();
	}


	public static void main(String args[]) throws IOException, InterruptedException {
		// example usage
		LTUDPLogger logger = new  LTUDPLogger("fooLog.txt", 10001);
		logger.start();
		//Thread.sleep(10 * (60 *1000)); // log messages for 10 minutes
		Thread.sleep(1 * (10 *1000)); // log messages for 10 minutes
		System.out.println("Stopping");
		logger.stopRunning();
	}
}
