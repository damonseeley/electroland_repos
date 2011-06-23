package net.electroland.skate.core;

/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import com.illposed.osc.*;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.io.IOException;

public class SoundController{

	private InetAddress address;		// machine running max/msp
	private OSCPortOut sender;			// osc out
	private OSCMessage msg;			// osc packet
	private Object args[];				// osc content
	private String ip;
	private int nodeID;				// incrementing sound ID
	public boolean audioEnabled;		// turns audio on/off
	public float gain = 1;				// default volume level
	public int clamp = 1;

	public SoundController(String _ip, int port){
		try{
			ip = _ip;
			address = InetAddress.getByName(ip);		// a bad address will throw traxess parsing errors when using send!
			sender = new OSCPortOut(address, port);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}
		nodeID = 0;
		//audioEnabled = Boolean.parseBoolean(ConnectionMain.properties.get("audio"));
	}

	public int newSoundNode(String filename, int x, int y, float gain, String comment){
		nodeID++;
		//send("simple instance"+nodeID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
		return nodeID;
	}

	public void updateSoundNode(int id, int x, int y, float gain){
		//send("simple instance"+nodeID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
	}


	public int globalSound(String soundFile, boolean loop, float gain, String comment) {
		nodeID++;
		//send("global instance"+soundIDToStart+" "+soundFile+" "+gain+" "+comment);
		return nodeID;
	}

	/*
	// no longer in use
	public void killSound(int soundIDToKill){
		send("stop instance"+soundIDToKill);
	}
	 */

	public void killSound(){
		//send("kill"); //???
	}


	private void send(String command){
		/*
			if(audioEnabled){
			args = new Object[1];
			args[0] = command;
			msg = new OSCMessage(ip, args);
			try {
				sender.send(msg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}*/
	}

}
