package net.electroland.enteractive.diagnostic.old;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.Vector;

public class LTUDPBroadcasterManager extends Thread {

	private Vector<LTUDPBroadcaster> broadcasters = new Vector<LTUDPBroadcaster>();
	private int numLTUBS;
	private InetAddress address;
	private int startPort;
	private int startIP;
	private DatagramSocket socket;
	private String fullPacketString;
	
	private int count;

	public LTUDPBroadcasterManager(int numLTUBS, int startPort, int startIP) {
		this.startPort = startPort;
		this.startIP = startIP;
		this.numLTUBS = numLTUBS;
		setupLTUBS();
		
	}
	
	public void setupLTUBS() {
		
		String IPPrefix = "192.168.0.";
		
		
		// hacky, creates one extra
		String fullIP = IPPrefix + startIP;
		LTUDPBroadcaster ltub1 = new LTUDPBroadcaster(startPort, fullIP);
		ltub1.start();
		broadcasters.add(ltub1);
		startPort++;
		
		for (int i=0;i<numLTUBS;i++) {
			fullIP = IPPrefix + startIP;
			LTUDPBroadcaster ltub = new LTUDPBroadcaster(startPort, fullIP);
			ltub.start();
			broadcasters.add(ltub);
			startPort++;
			startIP++;
		}
		
		System.out.println("Setup " + broadcasters.size() + " LTUDP Broadcasters");
	}
	
	/* pass thru methods */
	
	public void sendPacket(byte[] buf) {
		// send request
		//byte[] buf = new byte[256];
		//buf = thePacket.getBytes();
		
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).sendPacket(buf);
		}
		
	}
	
	public void setFramerate(int fr) {
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).setFramerate(fr);
		}
	}
	
	public void setPacketString(String ps){
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).setPacketString(ps);
		}
	}
	
	public void sendOne() {
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).sendOne();
		}
	}
	
	public void stopBroadcast() {
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).stopBroadcast();
		}
		//broadcast = false;
	}
	
	public void startBroadcast() {
		for (int i=0;i<broadcasters.size();i++) {
			broadcasters.get(i).startBroadcast();
		}
		//broadcast = true;
	}
	
	public void setIpAddress(String newIP) {
		// for now disabled
		//setupSocket(newIP);
	}

}
