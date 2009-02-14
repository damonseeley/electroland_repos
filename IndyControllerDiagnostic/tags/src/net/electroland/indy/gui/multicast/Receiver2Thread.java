package net.electroland.indy.gui.multicast;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;

import net.electroland.indy.test.Util;

/**
 * This class listens on a port, and outputs what it receives, with verbosity
 * defined by the GUI object that owns.  This is tightly coupled with the GUI 
 * for now.  Should create a Loggable interface instead, with methods for 
 * logTimesIsEnabled() and logReceivedPacketsIsEnabled() 
 * 
 * @author geilfuss
 *
 */
public class Receiver2Thread extends Thread {

	private int port;
	private IndyControllerDiagnostic runner;
	private boolean running = false;

	/**
	 * As mentioned above, runner should really be something like "Loggable"
	 * 
	 * @param port
	 * @param runner
	 */
	public Receiver2Thread(int port, IndyControllerDiagnostic runner){
		this.runner = runner;
		this.port = port;
	}

	public void start(){
		super.start();
		running = true;
	}

	public void stopClean(){
		running = false;
	}

	@Override
	public void run() {

		try {

			System.out.println("listening on " + port);
			DatagramSocket socket = new DatagramSocket(port);

			while(running){

				DatagramPacket p = new DatagramPacket(new byte[1000], 1000);

				socket.receive(p);

				byte[] receivedData = new byte[p.getLength()];
				System.arraycopy(p.getData(), 0, receivedData, 0, p.getLength());
				InetAddress rAddress = p.getAddress();

				// output the received packet (if logging for returned packets
				// is enabled)
				if (runner.logReceives.isSelected()){
					System.out.println(rAddress + ":" + port + 
										"\tUDP returned:\t" + 
							Util.bytesToHex(receivedData));
				}

				// since timing data is just superimposed on normal packets,
				// without any flag letting you know that it's in there, we do
				// this funny thing: try to generate a time stamp from the
				// first 12 bytes.  If the time stamp is within 10 seconds of
				// the current time, we assume it was intended to be a time
				// stamp.  Dirty.  i know.
				if (receivedData.length > 14 && runner.logTimes.isSelected()){

					byte[] timeBytes = new byte[13];// we're copying only 12 bytes below, but we allocate 13.  that properly leaves the first byte at zero.
					System.arraycopy(receivedData, 2, timeBytes, 1, 12);
					// hack. instead of sending all 13 bytes, we assume the
					// value of the first byte is good for the next very, very
					// long while.
					long sentTime = Util.decodeTime(timeBytes) + 1000000000000L;
					long time = System.currentTimeMillis() - sentTime;

					if (time < 10000 && time > 0){
						System.out.println(rAddress + ":" + port + 
											"\tUDP Packet took " + time + " ms.");
					}
				}
			}
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ArrayIndexOutOfBoundsException e){
			e.printStackTrace();
		}
	}
}