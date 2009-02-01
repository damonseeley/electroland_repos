package net.electroland.indy.gui.multicast;

import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.text.ParseException;

import net.electroland.indy.test.Util;

public class TCPSenderThread extends Thread {

	private IndyControllerDiagnostic runner;
	private boolean running = false;
	private Target2[] targets;
	private TCPReceiverThread[] responseProcessors;
	private Socket[] clients;
	
	public TCPSenderThread(IndyControllerDiagnostic runner, Target2[] targets){
		this.runner = runner;
		this.targets = targets;
	}

	public void start(){
		super.start();
		running = true;
	}

	public void stopClean(){

		running = false;
		if (clients != null){
			for (int i = 0; i < clients.length; i++){
				try {
					if (responseProcessors != null &&
						responseProcessors[i] != null)
						responseProcessors[i].stopClean();
					if (clients[i] != null)
						clients[i].close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	@Override
	public void run() {
		
		responseProcessors = new TCPReceiverThread[targets.length];
		clients = new Socket[targets.length];
		byte[] buffer = new byte[0];
		OutputStream out;
		int ctr = 0;
		int current = runner.byteSlider.getValue();
		int increment = 1;
		long d_send[] = new long[30];
		long lastSent = System.currentTimeMillis();
		int d_ptr = 1;
		
		while (running){

			if (runner.custom.isSelected()){
				// if custom packet entirely from text input,
				// EVERYTHING else gets ignored.
				
				//  BUG this should break out of the whole send loop and then change the state of the stream button.
				try {
					buffer = Util.hextToBytes(runner.customPacket.getText());
				} catch (NumberFormatException e) {
					e.printStackTrace();
				} catch (ParseException e) {
					e.printStackTrace();
				}

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
					
//				} else if (runner.stepThroughRecipients.isSelected()){
//					// trace pattern 
//					for (int i = 2; i < packetSize - 1; i++){
//						buffer[i] = (byte)0;
//					}
//					buffer[(ctr%(buffer.length-3)) + 2] = (byte)253;

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
			
			// check our connections.  regenerate any dead ones.
			for (int i = 0; i < clients.length; i++){
				
				try {
					if (clients[i] == null || clients[i].isClosed()){
//						System.out.println("reconnect " + targets[i]);
						clients[i] = new Socket(targets[i].getAddress(), targets[i].getPort());
					}
					if (responseProcessors[i] == null || !responseProcessors[i].isRunning()){
//						System.out.println("reconnect listener " + targets[i]);
						responseProcessors[i] = new TCPReceiverThread(clients[i].getInputStream(), targets[i], runner, clients[i]);
					}
					

					// any packet unique data should be written here.
					if (runner.includeTimeing.isSelected() && buffer.length > 14){
						sendTime = System.currentTimeMillis();
						byte[] sendTimeBytes = Util.encodeTime(sendTime);
						// note that we're intentionally truncating this
						// 13 digit number.  the first digit won't update
						// for a very, very long time.
						System.arraycopy(sendTimeBytes,1, buffer, 2, 12);
					}					
					
					// send the packet
					out = clients[i].getOutputStream();
					out.write(buffer);
					out.flush();

					if (runner.logSends.isSelected()){
						System.out.println(targets[i].getAddress() + ":" + 
								targets[i].getPort() + "\tTCP sent:\t" + 
								Util.bytesToHex(buffer));						
					}
					
				} catch (IOException e) {
					e.printStackTrace();
				}
			}


			try {
				// if we are sending one packet, not streaming.
				if (runner.streamButton.getText() == IndyControllerDiagnostic.START_STREAM){
					running = false;
				}else{
					Thread.sleep(1000/runner.fpsSlider.getValue());
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		//stopClean();
	}
}