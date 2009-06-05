package net.electroland.lafm.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Properties;

import net.electroland.scSoundControl.*;

public class SoundManager implements SCSoundControlNotifiable {

	private SCSoundControl ss;
	//private Process scsynthProcess;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private String absolutePath;
	private Properties systemProps;
	
	public SoundManager(int numOutputChannels, int numInputChannels, Properties systemProps){
		this.systemProps = systemProps;
		serverIsLive = false;
		soundFiles = new Hashtable<String, Integer>();
		absolutePath = systemProps.getProperty("soundPath");
		ss = new SCSoundControl(this);
		ss.init();
		ss.showDebugOutput(false);
		ss.set_serverResponseTimeout(5000);
	}

	public void loadBuffer(String soundFile){
		ss.readBuf(absolutePath+soundFile);
	}
	
	private int[] lookupCoordinatesForMax(int c){
		int[] channelCoords = new int[2];

		switch(c){
		case 1:
			channelCoords[0] = 1;
			channelCoords[1] = 1;
			break;
		case 2:
			channelCoords[0] = 1;
			channelCoords[1] = 2;
			break;
		case 3:
			channelCoords[0] = 2;
			channelCoords[1] = 1;
			break;
		case 4:
			channelCoords[0] = 2;
			channelCoords[1] = 2;
			break;
		case 5:
			channelCoords[0] = 3;
			channelCoords[1] = 1;
			break;
		case 6:
			channelCoords[0] = 3;
			channelCoords[1] = 2;
			break;
		}
		
		return channelCoords;
	}
	
	private float[] lookupCoordinatesForSC(int c, float gain){
		// this method needs a better way to switch between number
		// of channels and the distribution of amplitude values.
		float[] channelCoords = new float[6];

		switch(c){
		case 1:
			channelCoords[0] = gain;
			channelCoords[1] = 0;
			channelCoords[2] = 0;
			channelCoords[3] = 0;
			channelCoords[4] = 0;
			channelCoords[5] = 0;
			break;
		case 2:
			channelCoords[0] = 0;
			channelCoords[1] = gain;
			channelCoords[2] = 0;
			channelCoords[3] = 0;
			channelCoords[4] = 0;
			channelCoords[5] = 0;
			break;
		case 3:
			channelCoords[0] = 0;
			channelCoords[1] = 0;
			channelCoords[2] = gain;
			channelCoords[3] = 0;
			channelCoords[4] = 0;
			channelCoords[5] = 0;
			break;
		case 4:
			channelCoords[0] = 0;
			channelCoords[1] = 0;
			channelCoords[2] = 0;
			channelCoords[3] = gain;
			channelCoords[4] = 0;
			channelCoords[5] = 0;
			break;
		case 5:
			channelCoords[0] = 0;
			channelCoords[1] = 0;
			channelCoords[2] = 0;
			channelCoords[3] = 0;
			channelCoords[4] = gain;
			channelCoords[5] = 0;
			break;
		case 6:
			channelCoords[0] = 0;
			channelCoords[1] = 0;
			channelCoords[2] = 0;
			channelCoords[3] = 0;
			channelCoords[4] = 0;
			channelCoords[5] = gain;
			break;
		}
		
		return channelCoords;
	}
	
	public void parseSoundFiles(Properties sysProps){
		Iterator<Object> iter = sysProps.values().iterator();
		while(iter.hasNext()){
			String prop = (String)iter.next();
			String[] proplist = prop.split(",");
			for(int i=0; i<proplist.length; i++){
				if(proplist[i].endsWith(".wav")){
					if(!soundFiles.containsKey(proplist[i])){
						System.out.println(proplist[i]);	// print out sound file to make sure it's working
						soundFiles.put(absolutePath+proplist[i], -1);	// -1 default unassigned value
						loadBuffer(proplist[i]);
					}
				}
			}
		}
	}
	
	// this doesn't matter anymore
	public int newSoundID(){
		return 0;
	}
	
	// this is a replacement for the original playSimpleSound
	public SoundNode playSimpleSound(String filename, int c, float gain, String comment){
		if(!filename.equals("none") && serverIsLive){
			float[] amplitudes = lookupCoordinatesForSC(c, gain);
			return ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), false, amplitudes, 1.0f);
		}
		return null;
	}
	
	// this is a replacement for the original globalSound
	public SoundNode globalSound(int soundIDToStart, String filename, boolean loop, float gain, int duration, String comment) {
		if(!filename.equals("none") && serverIsLive){
			return ss.createStereoSoundNodeWithLRMap(soundFiles.get(absolutePath+filename), false, new int[]{1, 0}, new int[]{0, 1}, 1.0f);
		}
		return null;
	}
	
	public void receiveNotification_BufferLoaded(int id, String filename) {
		System.out.println("Loaded buffer " + id + ", " + filename);
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}
	}

	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		parseSoundFiles(systemProps);
	}

	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU) {
		// TODO Keep track of server load
	}

	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
	}

	public void receiveNotification_ServerStatus(float averageCPU,
			float peakCPU, int numSynths) {
		// TODO Auto-generated method stub
		
	}
	
	
	
	
	
	public static void main(String[] args) {	// PROGRAM LAUNCH FOR TESTING ONLY
		Properties systemProps = null;
		try{
			systemProps = new Properties();
			systemProps.load(new FileInputStream(new File("depends//system.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		SCSoundManager soundManager = new SCSoundManager(2,2);
		soundManager.parseSoundFiles(systemProps);
	}

}


// TODO: THIS IS THE ORIGINAL SOUND MANAGER FOR USE WITH MAX

//package net.electroland.lafm.core;
//
//import com.illposed.osc_ELmod.*;
//
//import java.net.InetAddress;
//import java.net.SocketException;
//import java.net.UnknownHostException;
//import java.io.IOException;
//
//public class SoundManager extends Thread {
//	
//	private InetAddress address;		// machine running max/msp
//	private OSCPortOut sender;			// osc out
//	private OSCMessage msg;			// osc packet
//	private Object args[];				// osc content
//	private String ip;
//	private int soundID;				// incrementing sound ID
//	public boolean audioEnabled = true;		// turns audio on/off
//	public int gain = 1;				// default volume level
//	public int clamp = 1;
//
//	public SoundManager(String ip, int port){
//		try{
//			this.ip = ip;
//			address = InetAddress.getByName(ip);		// a bad address will throw traxess parsing errors when using send!
//			sender = new OSCPortOut(address, port);
//		} catch (SocketException e){
//			System.err.println(e);
//		} catch (UnknownHostException e){
//			System.err.println(e);
//		}
//		soundID = 0;
//	}
//	
///*	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
//		System.out.println(filename);
//		if(!filename.equals("none")){
//			System.out.println(filename);
//			soundID++;
//			int[] speaker = getNearestSpeaker(x,y);
//			send("simple instance"+soundID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
//		}
//	}*/
//	
//	public void playSimpleSound(String filename, int c, float gain, String comment){ // c is channel number
//		// this version takes a channel as argument and sends the appropriate coords to Marc Nimoy's max patch
//		//System.out.println(filename);
//		if(!filename.equals("none")){
//			//System.out.println(filename);
//			soundID++;
//			int[] channelCoords = lookupCoordinates(c);
//			//System.out.println("Played sound on channel " +c);
//			send("simple instance"+soundID+" "+filename+" "+channelCoords[0]+" "+channelCoords[1]+" 0 "+gain+" "+comment);
//		}
//	}
//	
//	public void globalSound(int soundIDToStart, String filename, boolean loop, float gain, int duration, String comment) {
//		if(!filename.equals("none")){
//			// duration not used, no looping
//			
//			send("global instance"+soundIDToStart+" "+filename+" "+gain+" "+comment);
//			
//			// TO SEND FULL GAIN FOR MIXING TEST
//			//send("global instance"+soundIDToStart+" "+filename+" "+1.0+" "+comment);
//		}
//	}
//	
//	private int[] lookupCoordinates(int c){
//		int[] channelCoords = new int[2];
//
//		switch(c){
//		case 1:
//			channelCoords[0] = 1;
//			channelCoords[1] = 1;
//			break;
//		case 2:
//			channelCoords[0] = 1;
//			channelCoords[1] = 2;
//			break;
//		case 3:
//			channelCoords[0] = 2;
//			channelCoords[1] = 1;
//			break;
//		case 4:
//			channelCoords[0] = 2;
//			channelCoords[1] = 2;
//			break;
//		case 5:
//			channelCoords[0] = 3;
//			channelCoords[1] = 1;
//			break;
//		case 6:
//			channelCoords[0] = 3;
//			channelCoords[1] = 2;
//			break;
//		}
//		
//		return channelCoords;
//	}
//	
//	public int[] getNearestSpeaker(int x, int y){	// x/y are light locations
//		int[] speakerloc = new int[2];
//		speakerloc[0] = y/2;
//		if(speakerloc[0] < 1){
//			speakerloc[0] = 1;
//		} else if(speakerloc[0] > 12){
//			speakerloc[0] = 12;
//		}
//		if(x <= 2){
//			speakerloc[1] = 2;
//		} else {
//			speakerloc[1] = 1;
//		}
//		return speakerloc;
//	}
//	
//	public int newSoundID(){
//		soundID++;
//		return soundID;
//	}
//	
//	public void killAllSounds(){
//		send("kill");
//	}
//	
//	public String randomSound(String[] soundfiles){
//		
//		int filenumber = (int)(Math.random()*soundfiles.length);
//		if(filenumber == soundfiles.length){
//			filenumber--;
//		}
//		return soundfiles[filenumber];
//	}
//	
//	private void send(String command){
//		if(audioEnabled){
//			args = new Object[1];
//			args[0] = command;
//			msg = new OSCMessage(ip, args);
//			try {
//				sender.send(msg);
//			} catch (IOException e) {
//				System.err.println(e);
//			} 
//		}
//	}
//	
//}
