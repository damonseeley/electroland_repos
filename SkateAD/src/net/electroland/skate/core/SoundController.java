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
	
	public HashMap<Integer,SoundNode> soundNodes;

	public ChannelPool pool; 			// pool of available channels.
	
	private static Logger logger = Logger.getLogger(SoundController.class);

	public SoundController(String ip, int maxPort, int sesPort, int maxCh, Point2D.Double listenPos) {
		try{
			ipString = ip;
			ipAddress = InetAddress.getByName(ipString);		// a bad address will throw traxess parsing errors when using send!
			maxSender = new OSCPortOut(ipAddress, maxPort);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}
		
		try{
			ipString = ip;
			ipAddress = InetAddress.getByName(ipString);		// a bad address will throw traxess parsing errors when using send!
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
		soundNodes  = new HashMap<Integer,SoundNode>();
		pool = new ChannelPool(maxCh);
		setupListener();

	}

	public int newSoundNode(String soundFile, Point2D.Double pos, float gain){
		nodeID++;

		//send play command as SPATF
		String[] newSoundArgs = new String[1];
		// Ryan's patch only takes one number (node and/or channel baked into one)
		newSoundArgs[0] = nodeID + "";
		newSoundArgs[1] = soundFile;

		// --- bradley: replaced getNewChannel() with pool.getFirstAvailable();
		int newSoundChannel = pool.getFirstAvailable();
		logger.info("channel " + newSoundChannel + " assigned");

		if (newSoundChannel == -1){
			logger.info("Max->SES polyphony all used up - free up bus channels");
		} else {
			newSoundArgs[2] = newSoundChannel + "";
			sendToMax(newSoundArgs);
		}

		SoundNode soundNode = new SoundNode(nodeID,newSoundChannel, soundFile, 0);
		soundNodes.put(nodeID, soundNode);
		
		// update SES position
		updateSoundNode(nodeID,pos,1.0f);
		
		return nodeID;
		
		

	}

	public void updateSoundNode(int id, Point2D.Double skaterPos, float gain){
		/*
		 * update the location of soundNode nodeID in SES
		 * 
		 */

		double newTheta = computeAzimuth(audioListenerPos,skaterPos);
		//logger.info("Computed Theta = " + newTheta);
		double newDist = computeDistanceInMeters(audioListenerPos,skaterPos);
		int channelNum = soundNodes.get(id).soundChannel;
		
		String[] newPosArgs = new String[4];
		//Hmmmm, I think this should be a lookup of interbus channel instead
		newPosArgs[0] = channelNum + ""; // hacky way to convert int to string?
		newPosArgs[1] = newTheta + "";
		newPosArgs[2] = 0 + "";
		newPosArgs[3] = newDist + "";
		sendToSES(newPosArgs);

		
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
						float tmpAmp = Float.parseFloat(message.getArguments()[2].toString());
						setAmp(Integer.parseInt(message.getArguments()[1].toString()),tmpAmp);
					}

					if (message.getArguments()[0].toString().matches("bufEnd")) {
						// remove the soundNode from soundNodes
						logger.info("bufferEnd received for : " + message.getArguments()[1]);
						removeNode(Integer.parseInt(message.getArguments()[1].toString()));
					}
					
					/*
					//for checking whole messages
					for (Object i : message.getArguments()){
						System.out.print(i + " ");
					}
					System.out.println("");
					*/
					
					
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
	
	public void removeNode (int id){
		if (soundNodes.containsKey(id)) {
			pool.releaseChannel(soundNodes.get(id).soundChannel);
			soundNodes.remove(id);
			
		} else {
			logger.info("Tried to remove non-existent soundNode: " + id);
		}
	}
	
	public void setAmp(int id, float amp){
		if (soundNodes.containsKey(id)) {
			soundNodes.get(id).amplitude = amp;
		} else {
			logger.info("Tried to set amp value for non-existent soundNode: " + id);
		}
	}
	
	public float getAmp(int id) {
		// do some checking here to make sure it exists?
		return soundNodes.get(id).amplitude;
	}

	
	private void sendToMax(String args[]){

		if(SkateMain.audioEnabled){
			String command = "/Play";
			String argConcat = args[0];
			for (int i = 1; i<args.length; i++) {
				argConcat += " " + args[i];
			}
			logger.info(argConcat);
			
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
			String argConcat = "SpatDIF";
			for (int i = 0; i<args.length; i++) {
				argConcat += "/" + args[i];
			}
			//System.out.println(argConcat);

			Object argToSend[] = new Object[1];
			argToSend[0] = argConcat;
			sesMsg = new OSCMessage(ipString, argToSend);
			try {
				sesSender.send(sesMsg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
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