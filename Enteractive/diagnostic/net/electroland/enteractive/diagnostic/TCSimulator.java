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
	private static long curTime = System.currentTimeMillis(); // time of frame start to aviod lots of calls to System.getcurentTime()
	private static long elapsedTime = -1; // time between start of cur frame and last frame to avoid re calculating passage of time allover the place

	private boolean isRunning = true;
	
	private InetAddress address;
	private int port;
	private String ip;
	private DatagramSocket socket;

	
	private int count;

	public TCSimulator(String ip, int port) {
		this.port = port;
		this.ip = ip;
		
		framerate = 30;
		
		timer = new Timer(framerate);
		setupSocket(this.ip);
		
		//System.out.println();
		//logger.info("Setup TC Broadcaster with initial IP " + ip + " port " + this.port);
		
	}
	
	public void setupSocket(String newIP) {
		try {
			// TODO Auto-generated method stub
			address = InetAddress.getByName(newIP);
			//port = 10001;
			socket = new DatagramSocket();
		} catch (Exception e) {
			logger.error("bad address" + e);
			// TODO: handle exception
		}
		logger.debug("Broadcaster set socket properties to IP " + ip + " on port " + port);
	}

	public void run() {

		timer.start();
		curTime = System.currentTimeMillis();
		count = 0;

		while (isRunning) {
			
			long tmpTime = System.currentTimeMillis();
			elapsedTime = tmpTime - curTime;
			curTime = tmpTime;

			if (Math.random()*100 > 90){
				//randomly set some tiles on/off
				String tileStates = "";
				for (int i=0;i<7;i++){
					if (Math.random()*100 > 70){
						tileStates += "fd";
					} else {
						tileStates += "00";
					}
				}
				
				double offset = Math.random()*22;
				String offsetByteA = Integer.toString((int)Math.round(offset));
				if (offsetByteA.length() == 1){
					offsetByteA = "0" + offsetByteA;
				}
				logger.info(offsetByteA);
				
				String fullPacketString = "FF02" + tileStates + offsetByteA + "00" + "FE";
				byte[] b = HexUtils.hexToBytes(fullPacketString);
				sendPacket(b);
				logger.info("sent " + fullPacketString);
				//System.out.println("Sending buffer " + b);
			}

			timer.block();

		}
		socket.close();

	}
	
	public void sendPacket(byte[] buf) {

		DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
		//System.out.println(this + "    port: " + port);
		try {
			socket.send(packet);
//			if (count % 100 == 0) {
//				logger.info(count + " packets sent on port " + port);
//			}
//			count++;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			logger.info("Error on port " + port);
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
