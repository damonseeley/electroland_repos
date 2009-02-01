package net.electroland.enteractive.diagnostic.old;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

import net.electroland.enteractive.diagnostic.Timer;
import net.electroland.enteractive.utils.HexUtils;

public class LTUDPBroadcaster extends Thread {

	private float framerate;
	private Timer timer;
	private static long curTime = System.currentTimeMillis(); // time of frame start to aviod lots of calls to System.getcurentTime()
	private static long elapsedTime = -1; // time between start of cur frame and last frame to avoid re calculating passage of time allover the place

	private boolean isRunning = true;
	private boolean broadcast = false;
	
	private InetAddress address;
	private int port;
	private String ip;
	private DatagramSocket socket;
	private String fullPacketString;
	
	private int count;

	public LTUDPBroadcaster(int port, String ip) {
		this.port = port;
		this.ip = ip;

		framerate = 1;

		fullPacketString = "00";
		
		timer = new Timer(framerate);
		setupSocket(this.ip);
		
		System.out.println("Setup LTUDP Broadcaster IP " + ip + " port " + this.port);
		/*try {
			// TODO Auto-generated method stub
			address = InetAddress.getByName(ip);
			port = 10001;
			socket = new DatagramSocket();
		} catch (Exception e) {
			// TODO: handle exception
		}*/
		
	}
	
	public void setupSocket(String newIP) {
		try {
			// TODO Auto-generated method stub
			address = InetAddress.getByName(newIP);
			//port = 10001;
			socket = new DatagramSocket();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	public void run() {

		timer.start();
		curTime = System.currentTimeMillis();
		count = 0;

		while (isRunning) {
			if (broadcast) {
				long tmpTime = System.currentTimeMillis();
				elapsedTime = tmpTime - curTime;
				curTime = tmpTime;

				byte[] b = HexUtils.hexToBytes(fullPacketString);
				sendPacket(b);
				//System.out.println("Sending buffer " + b);
			}

			timer.block();

		}
		socket.close();

	}
	
	public void sendPacket(byte[] buf) {
		// send request
		//byte[] buf = new byte[256];
		//buf = thePacket.getBytes();
		DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
		//System.out.println(this + "    port: " + port);
		try {
			socket.send(packet);
			if (count % 100 == 0) {
				System.out.println(count + " packets sent on port " + port);
			}
			count++;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("Error on port " + port);
		}
	}
	
	public void setFramerate(int fr) {
		framerate = (float)fr;
		timer.setFrameRate(framerate);
		//System.out.println("framerate: " + framerate);
	}
	
	public void setPacketString(String ps){
		fullPacketString = ps;
	}
	
	public void sendOne() {
		byte[] b = HexUtils.hexToBytes(fullPacketString);
		sendPacket(b);
		System.out.println("Sent one: " + fullPacketString + " on port " + port);
	}
	
	public void stopBroadcast() {
		System.out.println("Stop Broadcast");
		broadcast = false;
	}
	
	public void startBroadcast() {
		System.out.println("Start Broadcast");
		broadcast = true;
	}
	
	public void setIpAddress(String newIP) {
		setupSocket(newIP);
	}

}
