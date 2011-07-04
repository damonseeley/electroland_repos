package net.electroland.skate.core;

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

import oscP5.OscMessage;
import oscP5.OscP5;
import processing.core.PApplet;

public class SoundControllerP5 {

	private InetAddress ipAddress;		// machine running max/msp
	private String ipString;			// string ip address incoming

	//OSCP5 stuff
	private OscP5 oscP5Max;
	private OscP5 oscP5SES;
	private NetAddress maxBroadcastLoc;
	private NetAddress sesBroadcastLoc;

	/*
	private OSCPortOut maxSender;			// osc out
	private OSCPortOut sesSender;			// osc out to SESs
	private OSCMessage maxMsg;				// osc packet
	private OSCMessage sesMsg;				// osc packet
	 */

	private int nodeID;					// incrementing sound ID
	public float gain = 1.0f;			// default volume level
	private Point2D.Double audioListenerPos = new Point2D.Double(10,10);

	public HashMap<Integer,SoundNode> soundNodesByID;
	public HashMap<Integer,SoundNode> soundNodesByChannel;

	public ChannelPool pool; 			// pool of available channels.

	private static Logger logger = Logger.getLogger(SoundControllerP5.class);

	public SoundControllerP5(String ip, int maxPort, int sesPort, int maxCh, Point2D.Double listenPos) {

		ipString = ip; // same IP for both objects since max and ses have to exist on one box
		//logger.info("IPSTRING = " + ipString);
		try {
			ipAddress = InetAddress.getByName(ipString);
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//logger.info("IP STRING for OSCP5: " + ipString);

		oscP5Max = new OscP5(this,11000);
		maxBroadcastLoc = new NetAddress(ipString, 10000);

		oscP5SES = new OscP5(this,19999); //listen on some arbitrary port here
		sesBroadcastLoc = new NetAddress(ipString, 7770);

		audioListenerPos.x = listenPos.x;
		audioListenerPos.y = listenPos.y;
		logger.info("LISTENPOS = " + audioListenerPos.x + ", INCOMING = " + listenPos.x);

		nodeID = 0;
		soundNodesByID  = new HashMap<Integer,SoundNode>();
		soundNodesByChannel  = new HashMap<Integer,SoundNode>();
		pool = new ChannelPool(maxCh);
		
		//setupListener();

	}

	/*
	 * REMOTELY CALLED METHODS
	 */

	public int newSoundNode(String soundFile, Point2D.Double pos, float gain){
		nodeID++;

		// Ryan's patch only takes one number (node and/or channel baked into one)
		int newSoundChannel = pool.getFirstAvailable();
		logger.info("CHANNEL " + newSoundChannel + " assigned");
		if (newSoundChannel == -1){
			logger.info("Max->SES polyphony all used up - free up bus channels");
		} else {
			sendToMax(soundFile, newSoundChannel);
		}

		//SUPER HACKY WAY OF KEEPING TRACK OF EVERYTHING RIGHT NOW
		SoundNode soundNode = new SoundNode(nodeID,newSoundChannel,soundFile,0); //id, soundChannel, file, amplitude value
		soundNodesByChannel.put(soundNode.soundChannel,soundNode);
		soundNodesByID.put(soundNode.nodeID, soundNode);
		//logger.info("Map sizes: byID: " + soundNodesByID.size() + " byChannel: " + soundNodesByChannel.size());

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

			sendToSES(channelNum, newTheta, newDist);
		} else {
			//logger.info("ERROR: Tried to update non-existent nodeID: " + id);
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

			sendToSES(channelNum, newTheta, newDist);

			// call this here to remove.  If this happens before a bufEnd call it's OK.
			removeNodeByID(id);			
		} else {
			//logger.info("ERROR: tried to deallocate non-existent nodeID: " + id);
		}

	}


	/*
	 * setup listener for incoming msg
	 */

	/* incoming osc message are forwarded to the oscEvent method. */
	void oscEvent(OscMessage msg) {

		String msgArgs = "";
		for (Object o : msg.arguments()){
			msgArgs += o + " ";
		}
		//logger.info("INCOMING OSC = " + msgArgs);


		if (msg.arguments()[0].toString().matches("amp")) {  //use matches instead
			// update the amplitude value for nodeID
			//logger.info(message.getArguments()[2].getClass());
			int channelToUpdate = Integer.parseInt(msg.arguments()[1].toString());
			float amp = Float.parseFloat(msg.arguments()[2].toString());

			setAmpByChannel(channelToUpdate,amp);
		}

		if (msg.arguments()[0].toString().matches("bufEnd")) {  //use matches instead
			// update the amplitude value for nodeID
			//logger.info(message.getArguments()[2].getClass());
			int channelEnded = Integer.parseInt(msg.arguments()[1].toString());

			removeNodeByChannel(channelEnded);
		}

	}


	/*
	 * Node and node map modification methods
	 */

	private void removeNodeByID (int id){
		if (soundNodesByID.containsKey(id)) {
			pool.releaseChannel(soundNodesByID.get(id).soundChannel);
			soundNodesByChannel.remove(soundNodesByID.get(id).soundChannel);
			soundNodesByID.remove(id);
			//logger.info("REMOVED soundNode by ID: " + id);
		} else {
			//logger.info("ERROR: Tried to remove non-existent soundNode id: " + id);
		}
	}

	private void removeNodeByChannel (int ch){
		if (soundNodesByChannel.containsKey(ch)) {
			pool.releaseChannel(soundNodesByChannel.get(ch).soundChannel);
			soundNodesByID.remove(soundNodesByChannel.get(ch).nodeID);
			soundNodesByChannel.remove(ch);
			logger.info("Removed soundNode by channel: " + ch);
		} else {
			//logger.info("ERROR: Tried to remove non-existent soundNode channel: " + ch);
		}
	}

	private void setAmpByID(int id, float amp){
		if (soundNodesByID.containsKey(id)) {
			soundNodesByID.get(id).amplitude = amp;
		} else {
			//logger.info("ERROR: Tried to set amp value for non-existent soundNodeByID: " + id);
		}
	}

	private void setAmpByChannel(int ch, float amp){
		if (soundNodesByChannel.containsKey(ch)) {
			soundNodesByChannel.get(ch).amplitude = amp;
		} else {
			//logger.info("ERROR: Tried to set amp value for non-existent soundNodeByChannel: " + ch);
		}
	}

	public float getAmpByID(int id) {
		// do some checking here to make sure it exists?
		if (soundNodesByID.containsKey(id)) {
			// normalize to -0.2 - 0.2
			float normalizedAmp = (soundNodesByID.get(id).amplitude + 0.2f)/0.4f;
			if (normalizedAmp < 0.0f){
				normalizedAmp = 0.0f;
			}
			return normalizedAmp;
		} else {
			return 0;
		}
	}


	private void sendToMax(String soundFile, int ch){

		if(SkateMain.audioEnabled){
			OscMessage oscMsg = new OscMessage("/Play");
			oscMsg.add(soundFile);
			oscMsg.add(ch);
			oscP5Max.send(oscMsg,maxBroadcastLoc);
			logger.info("SEND TO MAX: " + oscMsg.address() + " " + oscMsg.arguments()[0] + " " + oscMsg.arguments()[1]);
		}
	}

	private void sendToSES(int ch, double az, double dist){

		if(SkateMain.audioEnabled){
			// hacky custom string for SES
			/*
			OscMessage oscMsg = new OscMessage("/SpatDIF/source/" + ch +"/aed");
			oscMsg.add((float)roundTwoDec(az));
			oscMsg.add(0.0f);
			oscMsg.add((float)roundTwoDec(dist));
			 */

			OscMessage oscMsg = new OscMessage("/SES");
			oscMsg.add(ch);
			oscMsg.add((float)roundTwoDec(az));
			oscMsg.add((float)roundTwoDec(dist));
			oscP5SES.send(oscMsg,sesBroadcastLoc);
			//logger.info("SEND TO SES: " + oscMsg.address() + " " + oscMsg.arguments()[0] + " " + oscMsg.arguments()[1] + " " + oscMsg.arguments()[2]);

		}
	}

	//Rounding
	private double roundTwoDec(double d) {
		DecimalFormat twoDForm = new DecimalFormat("#.##");
		return Double.valueOf(twoDForm.format(d));
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
		//new SoundController("127.0.0.1",10000,7770,16,new Point2D.Double(0,0));
	}


}