package net.electroland.connection.core;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Properties;

import net.electroland.scSoundControl.SCSoundControl;
import net.electroland.scSoundControl.SCSoundControlNotifiable;
import net.electroland.scSoundControl.SoundNode;

public class SoundController implements SCSoundControlNotifiable {
	
	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private String absolutePath;
	private HashMap<String,String> systemProps;
	private int soundID;				// incrementing sound ID
	
	public boolean audioEnabled = true;
	
	public SoundController(int numOutputChannels, int numInputChannels, HashMap<String,String> systemProps){
		this.systemProps = systemProps;
		serverIsLive = false;
		soundFiles = new Hashtable<String, Integer>();
		absolutePath = (String) systemProps.get("soundPath");
		ss = new SCSoundControl(this);
		ss.init();
		ss.showDebugOutput(true);
		ss.set_serverResponseTimeout(5000);	
	}
	
	public void loadBuffer(String soundFile){
		ss.readBuf(absolutePath+soundFile);
	}
	
	public void parseSoundFiles(){
		Iterator<String> iter = systemProps.values().iterator();
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
	
	
	public float[] getAmplitudes(int x, int y, float gain){
		float[] amplitudes = new float[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		int channel = y/2;
		if(channel < 1){
			channel = 1;
		} else if(channel > 12){
			channel = 12;
		}
		if(x <= 2){
			channel = (channel*2) - 1;
		} else {
			channel = (channel*2);
		}
		channel -= 1; // for zero index
		amplitudes[channel] = gain;
		return amplitudes;
	}
	
	public void globalSound(int soundIDToStart, String soundFile, boolean loop, float gain, int duration, String comment){
		if(!soundFile.equals("none") && serverIsLive){
			int[] channels = new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
			float[] amplitudes = new float[]{gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain};
			SoundNode sn = ss.createStereoSoundNodeWithLRMap(soundFiles.get(absolutePath+soundFile), false, new int[]{1, 0}, new int[]{0, 1}, 1.0f);
			sn.setAmplitudes(channels, amplitudes);
		}
	}
	
	public int newSoundID(){	// no longer in use, but referenced by all animations
		soundID++;
		return soundID;
	}
	
	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
		soundID++;
		if(!filename.equals("none") && serverIsLive){
			float[] amplitudes = getAmplitudes(x, y, gain);
			ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), false, amplitudes, 1.0f);
		}
	}
	
	public String randomSound(String[] soundfiles){
		int filenumber = (int)(Math.random()*soundfiles.length);
		if(filenumber == soundfiles.length){
			filenumber--;
		}
		return soundfiles[filenumber];
	}

	public void receiveNotification_BufferLoaded(int id, String filename) {
		System.out.println("Loaded buffer " + id + ", " + filename);
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}
	}

	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		parseSoundFiles();
	}

	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU, int numSynths) {
		// keep track of server load		
	}

	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
	}
	
	public void killAllSounds(){
		ss.freeAllBuffers();		
	}
	

}





//TODO: THIS IS THE ORIGINAL SOUND MANAGER FOR USE WITH MAX

//import com.illposed.osc.*;
//import java.net.InetAddress;
//import java.net.SocketException;
//import java.net.UnknownHostException;
//import java.io.IOException;
//
//public class SoundController{
//	
//	private InetAddress address;		// machine running max/msp
//	private OSCPortOut sender;			// osc out
//	private OSCMessage msg;			// osc packet
//	private Object args[];				// osc content
//	private String ip;
//	private int soundID;				// incrementing sound ID
//	public boolean audioEnabled;		// turns audio on/off
//	public int gain = 1;				// default volume level
//	public int clamp = 1;
//
//	public SoundController(String _ip, int port){
//		try{
//			ip = _ip;
//			address = InetAddress.getByName(ip);		// a bad address will throw traxess parsing errors when using send!
//			sender = new OSCPortOut(address, port);
//		} catch (SocketException e){
//			System.err.println(e);
//		} catch (UnknownHostException e){
//			System.err.println(e);
//		}
//		soundID = 0;
//		audioEnabled = Boolean.parseBoolean(ConnectionMain.properties.get("audio"));
//	}
//	
//	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
//		soundID++;
//		int[] speaker = getNearestSpeaker(x,y);
//		send("simple instance"+soundID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
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
//	public void globalSound(int soundIDToStart, String soundFile, boolean loop, float gain, int duration, String comment) {
//		// duration not used, no looping
//		// send simple instanceID soundfilename.wav 0 0 1 1
//		send("global instance"+soundIDToStart+" "+soundFile+" "+gain+" "+comment);
//		//send("simple instance"+soundIDToStart+" "+soundFile+" "+0+" "+0+" "+0+" "+gain+" "+comment);
//	}
//	
//	/*
//	// no longer in use
//	public void killSound(int soundIDToKill){
//		send("stop instance"+soundIDToKill);
//	}
//	*/
//	
//	public void killAllSounds(){
//		send("kill");
//	}
//	
//	public String randomSound(String[] soundfiles){
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
