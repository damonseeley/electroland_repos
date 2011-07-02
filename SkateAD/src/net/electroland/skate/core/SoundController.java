package net.electroland.skate.core;

/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import java.awt.geom.Point2D;
import java.io.IOException;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.HashMap;

import org.apache.log4j.Logger;

import com.illposed.osc.OSCListener;
import com.illposed.osc.OSCMessage;
import com.illposed.osc.OSCPortIn;
import com.illposed.osc.OSCPortOut;

public class SoundController{

	private InetAddress ipAddress;		// machine running max/msp
	private OSCPortOut maxSender;			// osc out
	private OSCPortOut sesSender;			// osc out to SESs
	private OSCMessage maxMsg;				// osc packet
	private OSCMessage sesMsg;				// osc packet

	private Object args[];				// osc content
	private String ipString;			// string ip address incoming
	private int nodeID;					// incrementing sound ID
	public float gain = 1.0f;			// default volume level
	private Point2D.Double audioListenerPos = new Point2D.Double(10,10);

	public HashMap<Integer,SoundNode> soundNodesByID;
	public HashMap<Integer,SoundNode> soundNodesByChannel;

	public ChannelPool pool; 			// pool of available channels.

	private static Logger logger = Logger.getLogger(SoundController.class);

	public SoundController(String ip, int maxPort, int sesPort, int maxCh, Point2D.Double listenPos) {
		try{
			ipString = ip;
			logger.info("IPSTRING = " + ipString);
			ipAddress = InetAddress.getByName(ipString);		
			maxSender = new OSCPortOut(ipAddress, maxPort);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}

		try{
			ipString = ip;
			ipAddress = InetAddress.getByName(ipString);		
			sesSender = new OSCPortOut(ipAddress, sesPort);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}

		audioListenerPos.x = listenPos.x;
		audioListenerPos.y = listenPos.y;
		logger.info("LISTENPOS = " + audioListenerPos.x + ", INCOMING = " + listenPos.x);

		nodeID = 0;
		soundNodesByID  = new HashMap<Integer,SoundNode>();
		soundNodesByChannel  = new HashMap<Integer,SoundNode>();
		pool = new ChannelPool(maxCh);
		setupListener();

	}

	public int newSoundNode(String soundFile, Point2D.Double pos, float gain){
		nodeID++;

		String[] newSoundArgs = new String[2];
		// Ryan's patch only takes one number (node and/or channel baked into one)
		// newSoundArgs[0] = nodeID + "";  // all inter-app communication is now keyed to sound channel
		newSoundArgs[0] = soundFile;

		int newSoundChannel = pool.getFirstAvailable();
		logger.info("channel " + newSoundChannel + " assigned");
		if (newSoundChannel == -1){
			logger.info("Max->SES polyphony all used up - free up bus channels");
		} else {
			newSoundArgs[1] = newSoundChannel + "";
			sendToMax(newSoundArgs);
		}

		//SUPER HACKY WAY OF KEEPING TRACK OF EVERYTHING RIGHT NOW
		SoundNode soundNode = new SoundNode(nodeID,newSoundChannel,soundFile,0); //id, soundChannel, file, amplitude value
		soundNodesByChannel.put(soundNode.soundChannel,soundNode);
		soundNodesByID.put(soundNode.nodeID, soundNode);
		logger.info("Map sizes: byID: " + soundNodesByID.size() + " byChannel: " + soundNodesByChannel.size());

		// update SES position
		updateSoundNodeByID(nodeID,pos,1.0f);

		return nodeID;



	}

	// Update the location of soundNode nodeID in SES
	public void updateSoundNodeByID(int id, Point2D.Double skaterPos, float gain){

		if (soundNodesByID.containsKey(id)) {

			double newTheta = computeAzimuth(audioListenerPos,skaterPos);
			double newDist = computeDistanceInMeters(audioListenerPos,skaterPos);
			int channelNum = -1;

			/*
			 * Have a threadsafety issue here where it's possible that the OSCListener has removed
			 * the soundNode from both hashmaps before the skater has stopped animating (and sending update messages)
			 */
			try {
				channelNum = soundNodesByID.get(id).soundChannel;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			String[] newPosArgs = new String[4];
			//Hmmmm, I think this should be a lookup of interbus channel instead
			newPosArgs[0] = channelNum + ""; // hacky way to convert int to string?
			newPosArgs[1] = newTheta + "";
			newPosArgs[2] = 0 + "";
			newPosArgs[3] = newDist + "";
			sendToSES(newPosArgs);
		} else {
			logger.info("ERROR: tried to update non-existent nodeID: " + id + "  Probably thread-safe issue");
		}

	}

	// deallocate the soundChannel and 
	// ...update the location of soundNode nodeID in SES
	// this method is called by Skater once an Animation is complete
	// ... rather than being called from the sound system
	public void dellocateByID(int id, float gain){

		if (soundNodesByID.containsKey(id)) {

			double newTheta = 0;
			double newDist = 300; //300 meters, super far away.
			int channelNum = -1;

			/*
			 * Have a threadsafety issue here where it's possible that the OSCListener has removed
			 * the soundNode from both hashmaps before the skater has stopped animating (and sending update messages)
			 */
			try {
				channelNum = soundNodesByID.get(id).soundChannel;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			String[] newPosArgs = new String[4];
			//Hmmmm, I think this should be a lookup of interbus channel instead
			newPosArgs[0] = channelNum + ""; // hacky way to convert int to string?
			newPosArgs[1] = newTheta + "";
			newPosArgs[2] = 0 + "";
			newPosArgs[3] = newDist + "";
			sendToSES(newPosArgs);
			
			// call this here to remove.  If this happens before a bufEnd call it's OK.
			removeNodeByID(id);			
		} else {
			logger.info("ERROR: tried to deallocate non-existent nodeID: " + id);
		}

	}






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
					if (message.getArguments()[0].toString().matches("amp")) {  //use matches instead
						// update the amplitude value for nodeID
						//logger.info(message.getArguments()[2].getClass());
						int channelToUpdate = Integer.parseInt(message.getArguments()[1].toString());
						float amp = Float.parseFloat(message.getArguments()[2].toString());

						setAmpByChannel(channelToUpdate,amp);
					}

					if (message.getArguments()[0].toString().matches("bufEnd")) {
						// remove the soundNode from soundNodes
						logger.info("bufferEnd received for channel: " + message.getArguments()[1]);
						int channelToRemove = Integer.parseInt(message.getArguments()[1].toString());
						//int idToRemove = soundNodesByID;
						// should be deallocateByChannel here!!!!
						
						removeNodeByChannel(channelToRemove);
					}

					//printOSC(message);

				}
			};
			receiver.addListener("/skateamp", listener);
			receiver.addListener("/skaterdone", listener);
			receiver.startListening();
		} catch (SocketException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}




	public void removeNodeByID (int id){
		if (soundNodesByID.containsKey(id)) {
			pool.releaseChannel(soundNodesByID.get(id).soundChannel);
			soundNodesByChannel.remove(soundNodesByID.get(id).soundChannel);
			soundNodesByID.remove(id);
			logger.info("Removed soundNode by ID: " + id);
		} else {
			logger.info("Tried to remove non-existent soundNode id: " + id);
		}
	}

	public void removeNodeByChannel (int ch){
		if (soundNodesByChannel.containsKey(ch)) {
			pool.releaseChannel(soundNodesByChannel.get(ch).soundChannel);
			soundNodesByID.remove(soundNodesByChannel.get(ch).nodeID);
			soundNodesByChannel.remove(ch);
			logger.info("Removed soundNode by channel: " + ch);
		} else {
			logger.info("Tried to remove non-existent soundNode channel: " + ch);
		}
	}

	public void setAmpByID(int id, float amp){
		if (soundNodesByID.containsKey(id)) {
			soundNodesByID.get(id).amplitude = amp;
		} else {
			logger.info("Tried to set amp value for non-existent soundNodeByID: " + id);
		}
	}

	public void setAmpByChannel(int ch, float amp){
		if (soundNodesByChannel.containsKey(ch)) {
			soundNodesByChannel.get(ch).amplitude = amp;
		} else {
			logger.info("Tried to set amp value for non-existent soundNodeByChannel: " + ch);
		}
	}

	public float getAmpByID(int id) {
		// do some checking here to make sure it exists?
		return soundNodesByID.get(id).amplitude;
	}


	private void sendToMax(String args[]){

		if(SkateMain.audioEnabled){
			String command = "/Play";
			String argConcat = args[0];
			for (int i = 1; i<args.length; i++) {
				argConcat += " " + args[i];
			}
			//logger.info("SEND TO MAX: " + command + argConcat);

			Object argToSend[] = new Object[1];
			argToSend[0] = argConcat;
			//maxMsg = new OSCMessage(ipString, argToSend);
			maxMsg = new OSCMessage(command,argToSend);
			try {
				maxSender.send(maxMsg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}

	private void sendToSES(String args[]){

		if(SkateMain.audioEnabled){
			String command = "SpatDIF/";
			String argConcat = args[0];
			for (int i = 1; i<args.length; i++) {
				argConcat += "/" + args[i];
			}
			//logger.info("SEND TO SES: " + command + argConcat);

			Object argToSend[] = new Object[1];
			argToSend[0] = argConcat;
			sesMsg = new OSCMessage(command,argToSend);
			try {
				sesSender.send(sesMsg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}

	public void printOSC(OSCMessage msg){
		//for checking whole messages
		for (Object i : msg.getArguments()){
			System.out.print(i + " ");
		}
		System.out.println("");
	}



	/* 
	 * older stuff
	 */


	public int globalSound(String soundFile, boolean loop, float gain, String comment) {
		// not used now
		nodeID++;
		return nodeID;
	}

	public void killSound(){
		// not used now
	}


	public static void main(String[] args){
		new SoundController("127.0.0.1",10000,7770,16,new Point2D.Double(0,0));
	}

	// 2D (always assumes reference is "above" the listener)
	public static double computeAzimuth(Point2D.Double listener, 
			Point2D.Double object){

		// object in front (0) or behind user (180)
		if (listener.x == object.x){
			if (listener.y == object.y)
				return 0;
			//				return Double.NaN; // object on top of user.
			else
				return listener.y < object.y ? 180 : 0;
		}


		double radians = Math.atan2(object.y - listener.y, object.x - listener.x);
		double degrees = 90 + (180/Math.PI) * radians; 
		return degrees > 0 ? degrees : 360 + degrees;
	}

	// 2D
	public static double computeDistance(Point2D.Double listener, Point2D.Double object)
	{
		return listener.distance(object);
	}

	// 2D
	public static double computeDistanceInMeters(Point2D.Double listener, Point2D.Double object)
	{
		return listener.distance(object)/100;
	}
}