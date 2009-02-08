package net.electroland.enteractive.diagnostic;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;

import javax.swing.JTextArea;

import net.electroland.enteractive.diagnostic.old.LTOutputPanel;
import net.electroland.enteractive.utils.HexUtils;

/**
 * 	UDPReceiver just receives packets and sticks them on a msgQueue.  
 *  Be sure to call stopRunning() when done - this will close the socket properly.
 *
 * @author eitan
 *
 */
public class TCUDPReceiver extends Thread {
	boolean isRunning = true;

	public LinkedBlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>();

	private  DatagramSocket receiveSocket;

	private DatagramPacket receivePacket;

	int rcvPort;
	
	int count = 0;
	long lastTime = System.currentTimeMillis();
	long[] packetTimes = new long[10];
	
	//private LTOutputPanel ltto;
	private JTextArea outputTA;

	public TCUDPReceiver(int rcvPort) throws SocketException, UnknownHostException {
		receivePacket = new DatagramPacket(new byte[128], 64);
		this.rcvPort = rcvPort;
		receiveSocket = new DatagramSocket(rcvPort);
	}
	
	@SuppressWarnings("deprecation")
	public void setNewRcvPort(int newPort) throws SocketException, UnknownHostException {
		if (newPort != rcvPort) {
			rcvPort = newPort;
			isRunning = false;
			receiveSocket.close();
			receiveSocket = new DatagramSocket(rcvPort);
			isRunning = true;
			TCDiagnosticMain.logger.info("UDP reciever set to receive on port " + rcvPort);
		}
		
	}

	public void run() {
		//boolean reportedErr = false;
		try {
			// timeout and report error if not meesages from moderator in 2 secs
			receiveSocket.setSoTimeout(2000);
		} catch (SocketException e1) {
			// won't allow setting not really important
		}

		while (isRunning) {
			try {
				receiveSocket.receive(receivePacket);
				msgQueue.offer(new String(receivePacket.getData(), 0, receivePacket.getLength()));
				byte[] b = receivePacket.getData();
				//HexUtils.printHex(b);
				
				long thisTime = System.currentTimeMillis();
				long diff = thisTime - lastTime;
				lastTime = thisTime;
				packetTimes[count] = diff;
				if (count == 9) {
					count = 0;
				} else {
					count++;
				}
			
				//feedOutput(b, receivePacket.getLength());
				// roll the incoming text instead of replacing
				addOutput(b, receivePacket.getLength());
				
			} catch (SocketTimeoutException e) {
			} catch (IOException e) {
				TCDiagnosticMain.logger.info("IO Exception on UDP socket re-open.  Ignore for now");
				e.printStackTrace();
			}
		}
		receiveSocket.close();
	}
	
	public void registerOutput(TCOutputPanel ltto) {
		outputTA = ltto.getOutputField();
	}
	
	TCOutputPanel tcop;
	public void registerOutputPanel(TCOutputPanel tcop){
		this.tcop = tcop;
	}
	
	private void feedOutput(byte b[], int length) {
		//HexUtils.getBytesToHex(b);
		long intervalSum = 0;
		for (int a=0;a<packetTimes.length;a++) {
			intervalSum += packetTimes[a];
		}
		//outputTA.setText(HexUtils.bytesToHex(b,length) + "   interval= " + intervalSum/packetTimes.length);
		outputTA.setText(HexUtils.bytesToHex(b,length));
	}
	
	private void addOutput(byte b[], int length){
		//HexUtils.getBytesToHex(b);
		long intervalSum = 0;
		for (int a=0;a<packetTimes.length;a++) {
			intervalSum += packetTimes[a];
		}
		String newText = HexUtils.bytesToHex(b,length);
		tcop.updateTextField(newText);
	}

	public void stopRunning() {
		isRunning = false;
	}
}
