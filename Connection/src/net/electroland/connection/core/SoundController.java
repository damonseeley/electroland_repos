package net.electroland.connection.core;

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
	private int soundID;				// incrementing sound ID
	public boolean audioEnabled;		// turns audio on/off
	public int gain = 1;				// default volume level
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
		soundID = 0;
		audioEnabled = Boolean.parseBoolean(ConnectionMain.properties.get("audio"));
	}
	
	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
		soundID++;
		int[] speaker = getNearestSpeaker(x,y);
		send("simple instance"+soundID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
	}
	
	public int[] getNearestSpeaker(int x, int y){	// x/y are light locations
		int[] speakerloc = new int[2];
		speakerloc[0] = y/2;
		if(speakerloc[0] < 1){
			speakerloc[0] = 1;
		} else if(speakerloc[0] > 12){
			speakerloc[0] = 12;
		}
		if(x <= 2){
			speakerloc[1] = 2;
		} else {
			speakerloc[1] = 1;
		}
		return speakerloc;
	}
	
	public int newSoundID(){
		soundID++;
		return soundID;
	}
	
	public void globalSound(int soundIDToStart, String soundFile, boolean loop, float gain, int duration, String comment) {
		// duration not used, no looping
		// send simple instanceID soundfilename.wav 0 0 1 1
		send("global instance"+soundIDToStart+" "+soundFile+" "+gain+" "+comment);
		//send("simple instance"+soundIDToStart+" "+soundFile+" "+0+" "+0+" "+0+" "+gain+" "+comment);
	}
	
	public void killSound(int soundIDToKill){
		//send("stop instance"+soundIDToKill);
	}
	
	public void killAllSounds(){
		send("kill");
	}
	
	public String randomSound(String[] soundfiles){
		int filenumber = (int)(Math.random()*soundfiles.length);
		if(filenumber == soundfiles.length){
			filenumber--;
		}
		return soundfiles[filenumber];
	}
	
	private void send(String command){
		if(audioEnabled){
			args = new Object[1];
			args[0] = command;
			msg = new OSCMessage(ip, args);
			try {
				sender.send(msg);
			} catch (IOException e) {
				System.err.println(e);
			} 
		}
	}
	
}
