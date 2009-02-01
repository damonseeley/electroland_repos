package net.electroland.indy.gui.multicast;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.text.ParseException;

import net.electroland.indy.test.Util;

/**
 * This is the workhorse.  As mentioned in a few other places, this is currently
 * tightly coupled to MulticastRunner.  A better implementation would have some
 * interfaces to pass in for the Logging and Environment variables that it's
 * querying and updating from here.
 * 
 * @author geilfuss
 *
 */
public class Sender2Thread extends Thread {

	private Target2[] targets;
	private IndyControllerDiagnostic runner;
	private boolean running = false;
	private boolean single = false;
	private int seed = 0;

	/**
	 * Create a thread that sends packets to the specified targets at the whim
	 * of the runner, and using variables specified by the runner.
	 * 
	 * @param targets
	 * @param runner
	 * @param single - if true, send a single packet and die.
	 */
	public Sender2Thread(Target2[] targets, IndyControllerDiagnostic runner, boolean single){
		this.targets = targets;
		this.runner = runner;
		this.single = single;
	}

	/**
	 * Same as above, however lets you specify a seed value, that let's you
	 * continue a pattern you started with a previous thread (e.g, sync).  If
	 * you generate successive threads where single == true with seed
	 * incrementing by one for each successive thread, they'll provide the same
	 * behavior as if you'd called one single thread with single == false.
	 * 
	 * @param targets
	 * @param runner
	 * @param single
	 * @param seed
	 */
	public Sender2Thread(Target2[] targets, IndyControllerDiagnostic runner, boolean single, int seed){
		this.targets = targets;
		this.runner = runner;
		this.single = single;
		this.seed = seed;
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

		DatagramSocket socket;

		long d_send[] = new long[30];
		long lastSent = System.currentTimeMillis();
		int d_ptr = 1;
		byte[] buffer;
		int ctr = seed;
		
		try {
			socket = new DatagramSocket();

			while (running){
				
				if (runner.custom.isSelected()){
					// if custom packet entirely from text input,
					// EVERYTHING else gets ignored.
					buffer = Util.hextToBytes(runner.customPacket.getText());

				} else {

					// figure out how large our packet will be
					int packetSize = runner.pcktLengthSlider.getValue() + 3;

					// allocate the packet and set it's start, cmd, and end bytes
					buffer = new byte[packetSize];
					buffer[0] = (byte)255; // start byte
					buffer[1] = (byte)runner.cmdSlider.getValue();// command byte
					buffer[packetSize-1] = (byte)254; // end byte

					if (runner.ascending.isSelected()){
						// ascending
						for (int i = 2; i < packetSize -1 ; i++){
							buffer[i] = (byte)((i-2) % 253);
						}

					}else if (runner.oscillating.isSelected()){

						if (ctr%2 == 0){
							for (int i = 2; i < packetSize - 1; i++){
								buffer[i] = (byte)253;
							}
						}// else do nothing.  default is 'off' bytes.
						
					} else if (runner.stepThroughRecipients.isSelected()){
						// trace pattern 
						for (int i = 2; i < packetSize - 1; i++){
							buffer[i] = (byte)0;
						}
						buffer[(ctr%(buffer.length-3)) + 2] = (byte)253;

					}else{
						
						// packets that support complimentary bytes.
						
						if (runner.triangle.isSelected()){
							// triangle
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
							// update the display
							runner.byteSlider.setValue(current);

						}else if (runner.slider.isSelected()){
							// constant (from slider)
							current = runner.byteSlider.getValue();						
						}
						
						// if odds bytes are complimentary
						for (int i = 2; i < packetSize - 1; i++){
							if (runner.oddsCompliment.isSelected()){
								buffer[i] = i%2==0 ? (byte)(254 - current) : (byte)current;							
							}else{
								buffer[i] = (byte)current;
							}
						}
					}
				}

				// calculate fps					
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

				boolean isBroadcastTime = ctr == -1 ||
										  (runner.includeOffset.isSelected() && 
						  				   ctr%runner.offsetDelay.getValue() == 0);
				
				if (isBroadcastTime){
					byte[] o = {(byte)255, (byte)128, (byte)0, (byte)0, (byte)254};
					buffer = o;	
				}
				
				// send the packet to each target				
				for (int i = 0; i < targets.length; i++){
						
					if (isBroadcastTime){

						// copy the offset address for this target into
						// the packet
						
						System.arraycopy(targets[i].getOffset(), 0, buffer, 2, 2);
							
					}else if (runner.includeTimeing.isSelected() && buffer.length > 14){
						sendTime = System.currentTimeMillis();
						byte[] sendTimeBytes = Util.encodeTime(sendTime);
						// note that we're intentionally truncating this
						// 13 digit number.  the first digit won't update
						// for a very, very long time.
						System.arraycopy(sendTimeBytes,1, buffer, 2, 12);
					}

					DatagramPacket packet = new DatagramPacket(buffer, buffer.length,
																targets[i].getAddress(),
																targets[i].getPort());
					String toSend = Util.bytesToHex(buffer);
					if (runner.logSends.isSelected() && !isBroadcastTime)
						System.out.println(targets[i] + "\tUDP sent=     " + toSend);
					if (runner.logOffsets.isSelected() && isBroadcastTime)
						System.out.println(targets[i] + "\tUDP Offset sent=     " + toSend);

					try {
						socket.send(packet);
					} catch (IOException e) {
						System.out.println(targets[i]  + "\t" + e);
					}							
				}

				// sleep for the proper amount
				Thread.sleep(1000/runner.fpsSlider.getValue());
				
				if (single){
					running = false;
				}

				ctr++;

			}

		} catch (SocketException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		} finally {
			runner.streamButton.setSelected(false);
			runner.streamButton.setText(IndyControllerDiagnostic.START_STREAM);
			runner.oneButton.setEnabled(true);
		}
	}
}