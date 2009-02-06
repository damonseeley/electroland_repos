package net.electroland.lafm.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Properties;

import net.electroland.scSoundControl.*;

public class SCSoundManager implements SCSoundControlNotifiable {

	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	
	public SCSoundManager(int numOutputChannels, int numInputChannels){
		serverIsLive = false;
		soundFiles = new Hashtable<String, Integer>();
		//ss = new SCSoundControl(numOutputChannels, numInputChannels, this);
		//ss.init();
		//ss.showDebugOutput(true);
	}

	public void loadBuffer(String soundFile){
		ss.readBuf(soundFile);	// this needs to notify when loaded
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
						soundFiles.put(proplist[i], -1);	// -1 default unassigned value
						//loadBuffer(proplist[i]);
					}
				}
			}
		}
	}
	
	// this is a replacement for the original playSimpleSound
	public void playSimpleSound(String filename, int c, float gain, String comment){
		if(!filename.equals("none") && serverIsLive){
			float[] amplitudes = lookupCoordinatesForSC(c, gain);
			ss.createSoundNode(soundFiles.get(filename), false, amplitudes, 1.0f);
		}
	}

	// this is a replacement for the original globalSound
	public void globalSound(int soundIDToStart, String filename, boolean loop, float gain, int duration, String comment) {
		if(!filename.equals("none") && serverIsLive){
			ss.createSoundNode(soundFiles.get(filename), false, new float[] {gain,gain,gain,gain,gain,gain}, 1.0f);
		}
	}
	
	@Override
	public void receiveNotification_BufferLoaded(int id, String filename) {
		System.out.println("Loaded buffer " + id + ", " + filename);
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}
	}

	@Override
	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
	}

	@Override
	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU) {
		// TODO Keep track of server load
	}

	@Override
	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
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
