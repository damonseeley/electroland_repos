package net.electroland.skate.core;

/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import java.io.IOException;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Vector;

import com.illposed.osc.OSCListener;
import com.illposed.osc.OSCMessage;
import com.illposed.osc.OSCPort;
import com.illposed.osc.OSCPortIn;
import com.illposed.osc.OSCPortOut;

public class SoundController{

	private InetAddress ipAddress;		// machine running max/msp
	private OSCPortOut sender;			// osc out
	private OSCMessage msg;				// osc packet
	private Object args[];				// osc content
	private String ipString;			// string ip address incoming
	private int nodeID;					// incrementing sound ID
	public float gain = 1.0f;			// default volume level

	public SoundController(String ip, int port) {
		try{
			ipString = ip;
			ipAddress = InetAddress.getByName(ipString);		// a bad address will throw traxess parsing errors when using send!
			sender = new OSCPortOut(ipAddress, port);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}
		nodeID = 0;	
		
		/*
		 * setup listener
		 */
		OSCPortIn receiver;
		try {
			//receiver = new OSCPortIn(OSCPort.defaultSCOSCPort());
			receiver = new OSCPortIn(57130);
			OSCListener listener = new OSCListener() {
				public void acceptMessage(java.util.Date time, OSCMessage message) {
					
					for (Object i : message.getArguments()){
						System.out.println(i);
					}
					
				}
			};
			receiver.addListener("/skateapp", listener);
			receiver.startListening();
		} catch (SocketException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

	public Vector<String> soundNodes = new Vector<String>();
	
	public int newSoundNode(String filename, int x, int y, float gain, String comment){
		
		/*
		 * some pcode
		 * increment nodeID
		 * Create a soundNode and create a local object in a vector, 
		 * Tell SES where to place the node in space FIRST to avoid pops

		 * Tell max to start playing a sound, include a nodeID for callback, max returns a node ID?
		 * max starts playing and feeds the nodeID to an OSC return object
		 * 
		 * 
		 * back in java - the OSC listener receives amplitude messages from max and parses out to the node ID in the vector
		 * 
		 */
		//nodeID++;
		//send("simple instance"+nodeID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
		return nodeID;
	}

	public void updateSoundNode(int id, int x, int y, float gain){
		/*
		 * update the location of soundNode nodeID in SES
		 * 
		 */		//send("simple instance"+nodeID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
	}


	public int globalSound(String soundFile, boolean loop, float gain, String comment) {
		nodeID++;
		//send("global instance"+soundIDToStart+" "+soundFile+" "+gain+" "+comment);
		return nodeID;
	}


	public void killSound(){
		//send("kill"); //???
	}


	private void send(String command){

		if(SkateMain.audioEnabled){
			args = new Object[1];
			args[0] = command;
			msg = new OSCMessage(ipString, args);
			try {
				sender.send(msg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}

}
