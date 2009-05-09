package net.electroland.enteractive.diagnostic;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

import org.apache.log4j.Logger;

import net.electroland.enteractive.diagnostic.Timer;
import net.electroland.enteractive.utils.*;

public class TCSimulator extends Thread {

	static Logger logger = Logger.getLogger(TCSimulator.class);

	private float framerate;
	private Timer timer;

	private boolean isRunning = true;
	
	private InetAddress address;
	private int port;
	private String ip;
	private DatagramSocket socket;

	public TCSimulator(String ip, int port) {
		this.port = port;
		this.ip = ip;

		framerate = 30;

		timer = new Timer(framerate);
		setupSocket(this.ip);
		logger.info("Setup TC Broadcaster with initial IP " + ip + " port " + this.port);
	}

	public void setupSocket(String newIP) {
		try {
			address = InetAddress.getByName(newIP);
			socket = new DatagramSocket();
		} catch (Exception e) {
			logger.error("bad address" + e);
		}
		logger.info("Broadcaster set socket properties to IP " + ip + " on port " + port);
	}

	public void run() {

		timer.start();

		while (isRunning) {

			if (Math.random()*100 > 90){
				//randomly set some tiles on/off
				String tileStates = "";
				for (int i=0;i<8;i++){
					if (Math.random()*100 > 70){
						tileStates += "fd";
					} else {
						tileStates += "00";
					}
				}

				double offset = (Math.floor(Math.random()*22))*8;
				String offsetByteA = Integer.toString((int)Math.round(offset));
				if (offsetByteA.length() == 1){
					offsetByteA = "0" + offsetByteA;
				}
				logger.debug(offsetByteA);

				String fullPacketString = "FF20" + tileStates + HexUtils.decimalToHex(Integer.parseInt(offsetByteA)) + "00" + "FE";
				byte[] b = HexUtils.hexToBytes(fullPacketString);
				sendPacket(b);
				logger.debug("sent " + fullPacketString);
			}
			timer.block();
		}
		socket.close();
	}

	public void sendPacket(byte[] buf) {

		DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
		try {
			socket.send(packet);
		} catch (IOException e) {
			logger.error("Error on port " + port);
		}
	}

	public void setFramerate(int fr) {
		framerate = (float)fr;
		timer.setFrameRate(framerate);
	}

	public void sendOne(String stringToSend) {
		byte[] b = HexUtils.hexToBytes(stringToSend);
		sendPacket(b);
	}

	public void setNewSocketParameters(String newIP, int newPort) {
		socket.close();
		port = newPort;
		ip = newIP;
		setupSocket(ip);
	}

	public static void main(String[] args) {
		// change these args to have the simulator send to a different address/port
		TCSimulator tcs = new TCSimulator("localhost",10011);
		tcs.start();
	}
}