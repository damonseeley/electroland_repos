package net.electroland.indy.test;

import java.io.IOException;
import java.math.BigInteger;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

public class SenderThread extends Thread {

	private long sleepMillis;
	private Target[] targets;
	private Timer timer;

	public SenderThread(int fps, Target[] targets){

		this.targets = targets;

		// should really throw an exception if sleepMillis is negative.
		sleepMillis = fps < 0 ? 0 : 1000 / fps;

		System.out.println("Sender will send every " + sleepMillis + " millis to:");
		for (int i = 0; i < targets.length; i++){
			System.out.println(targets[i].getAddress() + ":" + targets[i].getPort());
		}
		System.out.println("--");

		timer = new Timer(fps);
	}
	
	@Override
	public void run() {

		// runs a triangle wave function between 0 and 253, using increment
		// as the slope
		int current = 0;
		int increment = 1;

		// static sections of the packet
		byte[] buffer = new byte[15];
		buffer[0] = (byte)255; // start byte
		buffer[1] = (byte)1; // command byte		
		buffer[14] = (byte)254; // end byte

		DatagramSocket socket;

		timer.start();

// for calculating FPS
//		long d_send[] = new long[30];
//		long lastSent = System.currentTimeMillis();
//		int d_ptr = 1;
		
		try {
			socket = new DatagramSocket();

			while (true){

				// triangle wave
				current += increment;
				
				if (current > 252){ // don't use start or end byte.
					increment *= -1;
					current = 253;
				}
				if (current < 1){
					increment *= -1;
					current = 2;
				}

				// data bytes
//				for (int i = 2; i < 8; i++){ // this is for storing the time
				for (int i = 2; i < 14; i++){
					if (i%2==0){
						buffer[i] = (byte)current;						
					}else{
						buffer[i] = (byte)(254 - current);
					}
				}
				
				boolean badpacket = false;
				// send the packet to each target
				for (int i = 0; i < targets.length; i++){

					// store the current time in the packet					
					long sendTime = System.currentTimeMillis();

//					d_send[d_ptr++%30] = sendTime - lastSent;
//					lastSent = sendTime;
//					long avg = 0;
//					for (int foo = 0; foo < d_send.length; foo++){
//						avg += d_send[foo];
//					}
//					avg = avg / d_send.length;
//					if (d_ptr%30==0) // print the average sleep
//						System.out.println("average=" + avg);

					
					byte[] t = BigInteger.valueOf(sendTime).toByteArray();
					for (int c = 0; c < t.length; c++){
						if (t[c] == (byte)255 || t[c] == (byte)254){
							badpacket = true;
						}
					}
//					System.arraycopy(t, 0, buffer, 8, 6);       // this actually writes the time.

					DatagramPacket packet = new DatagramPacket(buffer, buffer.length,
																targets[i].address,
																targets[i].port);
					if (badpacket){
//						System.out.println("Did not send a packet because it contained 0xFE or 0xFF");
					}else{
						
						// diagnostic: print this packet:	   				
						System.out.println("Packet sent=     " + Util.bytesToHex(buffer));
						socket.send(packet);
					}
				}

				// sleep for the proper amount
				timer.block();
			}		

		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally{
			// should close the socket and other cleanup here.
		}
	}
}