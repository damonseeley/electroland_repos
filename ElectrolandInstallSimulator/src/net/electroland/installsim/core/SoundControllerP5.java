package net.electroland.installsim.core;


/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import java.awt.geom.Point2D;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.text.DecimalFormat;
import java.util.HashMap;

import netP5.NetAddress;

import org.apache.log4j.Logger;

import oscP5.*;

public class SoundControllerP5 {

	private InetAddress ipAddress;		// machine running max/msp
	private String ipString;			// string ip address incoming

	//OSCP5 stuff
	private OscP5 oscP5Max;
	private NetAddress maxBroadcastLoc;


	private int nodeID;					// incrementing sound ID
	public float gain = 1.0f;			// default volume level

	private static Logger logger = Logger.getLogger(SoundControllerP5.class);

	public SoundControllerP5(String ip) {

		ipString = ip; // same IP for both objects since max and ses have to exist on one box
		//logger.info("IPSTRING = " + ipString);
		try {
			ipAddress = InetAddress.getByName(ipString);
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//logger.info("IP STRING for OSCP5: " + ipString);

		//listening on 11000
		oscP5Max = new OscP5(this,11000);
		maxBroadcastLoc = new NetAddress(ipString, 10000);
		
		//setupListener();

	}

	/*
	 * REMOTELY CALLED METHODS
	 */
	
	public void iStateUpdate(int id) {
		OscMessage oscMsg = new OscMessage("/ChOn");
		oscMsg.add(id);
		oscP5Max.send(oscMsg,maxBroadcastLoc);
		//logger.info("MAX KILL: " + oscMsg.arguments()[0]);
	}
	
	

	
	/**
	 * main()
	 */

	public static void main(String[] args){
		//new SoundController("127.0.0.1",10000,7770,16,new Point2D.Double(0,0));
	}


}