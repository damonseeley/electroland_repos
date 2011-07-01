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
	private OSCMessage msg;				// osc packet
	private Object args[];				// osc content
	private String ipString;			// string ip address incoming
	private int nodeID;					// incrementing sound ID
	public float gain = 1.0f;			// default volume level
	Point2D.Double listenerPos;
	
	public HashMap<Integer,SoundNode> soundNodes;

	public ChannelPool pool; 			// pool of available channels.
	
	// --- bradley: corrected reference to get the logger of this class here.
	private static Logger logger = Logger.getLogger(SoundController.class);

	public SoundController(String ip, int maxPort, int sesPort, int maxCh, double listenX, double listenY) {
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
			sesSender = new OSCPortOut(ipAddress, maxPort);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}
		
		listenerPos = new Point2D.Double(listenX,listenY);

		nodeID = 0;
		soundNodes  = new HashMap<Integer,SoundNode>();
		pool = new ChannelPool(maxCh);
		setupListener();

	}

	public int newSoundNode(String soundFile, Point2D.Double pos, float gain){
		nodeID++;

		//send play command as SPATF
		String[] newSoundArgs = new String[3];
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
		
		// update SES position
		updateSoundNode(nodeID,pos,1.0f);
		
		
		SoundNode soundNode = new SoundNode(nodeID,newSoundChannel, soundFile, 0);
		soundNodes.put(nodeID, soundNode);
		return nodeID;

	}

	public void updateSoundNode(int id, Point2D.Double skaterPt, float gain){
		/*
		 * update the location of soundNode nodeID in SES
		 * 
		 */
		
		double newTheta = computeAzimuth(listenerPos,skaterPt);
		double newDist = computeDistance(listenerPos,skaterPt);
		
		
		String[] newPosArgs = new String[4];
		//Hmmmm, I think this should be a lookup of interbus channel instead
		newPosArgs[0] = id + ""; // hacky way to convert int to string?
		newPosArgs[1] = (int) newTheta + "";
		newPosArgs[2] = 0 + "";
		newPosArgs[3] = (int) newDist + "";
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
						int tmpAmp = Integer.parseInt(message.getArguments()[2].toString());
						//soundNodes.get(message.getArguments()[1]).amplitude = tmpAmp;
						setAmp(Integer.parseInt(message.getArguments()[1].toString()),tmpAmp);
						//logger.info(soundNodes.get(message.getArguments()[1]).amplitude);
					}

					if (message.getArguments()[0].toString().matches("bufferEnd")) {
						//do some stuff related to the buffer ending
						// remove the soundNode from soundNodes
						logger.info("bufferEnd received for : " + message.getArguments()[1]);
						removeNode(Integer.parseInt(message.getArguments()[1].toString()));
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
	
	public void removeNode (int id){
		if (soundNodes.containsKey(id)) {
			soundNodes.remove(id);
		} else {
			logger.info("Tried to remove non-existent soundNode: " + id);
		}
	}
	
	public void setAmp(int id, int amp){
		if (soundNodes.containsKey(id)) {
			soundNodes.get(id).amplitude = amp;
		} else {
			logger.info("Tried to set amp value for non-existent soundNode: " + id);
		}
	}
	
	public int getAmp(int id) {
		// do some checking here to make sure it exists?
		return soundNodes.get(id).amplitude;
	}

	
	private void sendToMax(String args[]){

		if(SkateMain.audioEnabled){
			String argConcat = "/skateapp";
			for (int i = 0; i<args.length; i++) {
				argConcat += "/" + args[i];
			}
			
			Object argToSend[] = new Object[1];
			argToSend[0] = argConcat;
			msg = new OSCMessage(ipString, args);
			try {
				maxSender.send(msg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}

	private void sendToSES(String args[]){

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
				maxSender.send(msg);
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
		new SoundController("127.0.0.1",10000,7770,16,0,0);
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

		// translate so listener is on origin
		object.x -= listener.x;	listener.x = 0.0;
		object.y -= listener.y;	listener.y = 0.0;
		
		double radians = Math.atan2(object.y, object.x);
		return 90 + (180/Math.PI) * radians;
	}
	
	// 2D
	public static double computeDistance(Point2D.Double listener, Point2D.Double object)
	{
		return listener.distance(object);
	}
}