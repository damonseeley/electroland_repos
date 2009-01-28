package net.electroland.indy.gui.multicast;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;

import net.electroland.indy.test.Util;

public class TCPReceiverThread extends Thread {

	private BufferedInputStream br;
	private InputStream is;
	private boolean running = false;
	private Target2 target;
	private MulticastRunner runner;
	private Socket socket;

	public TCPReceiverThread(InputStream is, Target2 target, MulticastRunner runner, Socket socket){

		System.out.println("Receiver listening on port " + target.getPort());

		this.br = new BufferedInputStream(is);
		this.is = is;
		this.running = true;
		this.target = target;
		this.runner = runner;
		this.socket = socket;
		this.start();
	}

	public boolean isRunning(){
		return running;
	}

	public void stopClean(){
		running = false;
		try {
			if (br != null)
				br.close();
		} catch (IOException e) {
			// do nothing.
		}
	}
	
	@Override
	public void run() {
		
		try {
			while (running){
				
				StringBuffer sb = new StringBuffer();
				int available = br.available();
				if (available > 0){
					
					byte[] b = new byte[available]; 
					br.read(b);

					if (b.length > 14 && runner.logTimes.isSelected()){

						byte[] timeBytes = new byte[13];// we're copying only 12 bytes below, but we allocate 13.  that properly leaves the first byte at zero.
						System.arraycopy(b, 2, timeBytes, 1, 12);
						// hack. instead of sending all 13 bytes, we assume the
						// value of the first byte is good for the next very, very
						// long while.
						long sentTime = Util.decodeTime(timeBytes) + 1000000000000L;
						long time = System.currentTimeMillis() - sentTime;

						if (time < 10000 && time > 0){
							System.out.println(target + 
												"\tTCP Packet took " + time + " ms.");
						}
					
					
					}
					if (runner.logReceives.isSelected()){							
						System.out.println(target + "\tTCP received:\t" + Util.bytesToHex(b));
					}

				}
			}
		} catch (IOException e) {
			e.printStackTrace();
			stopClean();
		}
	}
}