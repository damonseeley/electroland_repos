package net.electroland.indy.gui.multicast;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

import net.electroland.indy.test.Target;
import net.electroland.indy.test.Util;

public class SliderThread extends Thread {

	private Target[] targets;
	private GUITestRunner runner;
	private boolean running = false;
	private boolean single = false;
	
	public SliderThread(Target[] targets, GUITestRunner runner, boolean single){
		this.targets = targets;
		this.runner = runner;
		this.single = single;
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

		int current = runner.byteSlider.getValue();
		int increment = 1;

		// static sections of the packet
		byte[] buffer = new byte[15];
		buffer[0] = (byte)255; // start byte
		buffer[14] = (byte)254; // end byte

		DatagramSocket socket;

		long d_send[] = new long[30];
		long lastSent = System.currentTimeMillis();
		int d_ptr = 1;
		
		try {
			socket = new DatagramSocket();

			while (running){

				buffer[1] = (byte)runner.cmdSlider.getValue();// command byte		
				
				if (!single && runner.triangle.isSelected()){
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
					
					runner.byteSlider.setValue(current);
					
				}else{
					current = runner.byteSlider.getValue();
				}				


				
				// data bytes
				for (int i = 2; i < 14; i++){
					if (runner.oddsCompliment.isSelected() && i%2==0){
						buffer[i] = (byte)(254 - current);
					}else{
						buffer[i] = (byte)current;												
					}

				}
				
				// send the packet to each target
				for (int i = 0; i < targets.length; i++){

					// store the current time in the packet					
					long sendTime = System.currentTimeMillis();

					d_send[d_ptr++%30] = sendTime - lastSent;
					lastSent = sendTime;
					long avg = 0;
					for (int foo = 0; foo < d_send.length; foo++){
						avg += d_send[foo];
					}
					avg = avg / d_send.length;

					if (avg != 0)
						runner.fps.setText("" + (1000 / avg));

					DatagramPacket packet = new DatagramPacket(buffer, buffer.length,
																targets[i].getAddress(),
																targets[i].getPort());

					System.out.println(targets[i] + " Packet sent=     " + Util.bytesToHex(buffer));
					socket.send(packet);
				}

				// sleep for the proper amount
				Thread.sleep(1000/runner.fpsSlider.getValue());
				
				if (single){
					running = false;
				}
			}

		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}