package net.electroland.skate.core;

/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import java.io.IOException;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.Vector;

import org.apache.log4j.Logger;

import com.illposed.osc.OSCListener;
import com.illposed.osc.OSCMessage;
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
	//public int maxChannels;				// the max channels for SES operations
	//public HashMap soundChUsed;			// hashmap of currently used channels where true means in-use
	public ChannelPool pool; 			// pool of available channels.

	private static Logger logger = Logger.getLogger(SoundController.class);

	public SoundController(String ip, int port, int maxCh) {
		try{
			ipString = ip;
			ipAddress = InetAddress.getByName(ipString);		// a bad address will throw traxess parsing errors when using send!
			sender = new OSCPortOut(ipAddress, port);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}

		pool = new ChannelPool(maxCh);
//		maxChannels = maxCh;
//		soundChUsed  = new HashMap();
//		for (int i = 1; i<=maxChannels; i++) { // start at 1 to reflect real world sound channels
//			soundChUsed.put(i, false); 
//		}
//		logger.info("soundChannels hashMap : " + soundChUsed);

		nodeID = 0;

		setupListener();

	}

	// Bradley: this returns the first unused channel.
//	private int getNewChannel() {
//		for (int i=1; i<=maxChannels; i++){
//			if ((Boolean)soundChUsed.get(i) == false){
//				soundChUsed.put(i, true);
//				return i;
//			} 
//		}
//		return -1;
//	}


	/*
	 * setup listener for incoming msg
	 */
	private void setupListener() {
		OSCPortIn receiver;
		try {
			receiver = new OSCPortIn(11000);
			OSCListener listener = new OSCListener() {
				public void acceptMessage(java.util.Date time, OSCMessage message) {

					// Parse through skaterlist and update amplitude
					// for some reason I can't do a simple == string compare here for conditional
					
					// Bradley: == is checks to see if they are the same object, which they aren't.
					// match checks to see if they are too different Strings that contain the same data.
					
					if (message.getArguments()[0].toString().matches("amplitude")) {  //use matches instead

						for (Skater sk8r : SkateMain.skaters) {
							if (sk8r.soundNode == Integer.parseInt(message.getArguments()[1].toString())) {
								//System.out.println("updating incoming skater amplitude");
								sk8r.amplitude = Integer.parseInt(message.getArguments()[2].toString());
							}
						}
					}

					if (message.getArguments()[0].toString().matches("bufferEnd")) {
						//do some stuff related to the buffer ending
					}

					/* for checking whole messages
					for (Object i : message.getArguments()){
						gSystem.out.print(i + " ");
					}
					System.out.println("");
					 */


				}
			};
			receiver.addListener("/skateapp", listener);
			receiver.startListening();
		} catch (SocketException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}


	public int newSoundNode(String filename, int x, int y, float gain, String soundFile){
		nodeID++;

		//send play command as SPATF
		String[] newSoundArgs = new String[3];
		newSoundArgs[0] = nodeID + "";
		newSoundArgs[1] = soundFile;
//		int newSoundChannel = getNewChannel();
		int newSoundChannel = pool.getFirstAvailable();
		if (newSoundChannel == -1){
			logger.info("Max->SES polyphony all used up - free up bus channels");
		} else {
			newSoundArgs[2] = newSoundChannel + "";
			sendSPATF(newSoundArgs);
		}

		return nodeID;

	}

	public void updateSoundNode(int id, int x, int y, float gain){
		/*
		 * update the location of soundNode nodeID in SES
		 * 
		 */
		String[] newPosArgs = new String[3];
		//Hmmmm, I think this should be a lookup of interbus channel instead
		newPosArgs[0] = nodeID + ""; // hacky way to convert int to string?
		newPosArgs[1] = x + "";
		newPosArgs[2] = y + "";
		sendSPATF(newPosArgs);
		
	}


	public int globalSound(String soundFile, boolean loop, float gain, String comment) {
		// not used now
		nodeID++;
		return nodeID;
	}


	public void killSound(){
		// to do
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

	private void sendSPATF(String args[]){

		if(SkateMain.audioEnabled){
			String argConcat = "SPATF";
			for (int i = 0; i<args.length; i++) {
				argConcat += "/" + args[i];
			}
			//System.out.println(argConcat);

			Object argToSend[] = new Object[1];
			argToSend[0] = argConcat;
			msg = new OSCMessage(ipString, args);
			try {
				sender.send(msg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}

	public static void main(String[] args){
		new SoundController("127.0.0.1",8888,16);
	}


}


