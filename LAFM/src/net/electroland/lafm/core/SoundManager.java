package net.electroland.lafm.core;

import com.illposed.osc.*;

import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.io.IOException;

public class SoundManager extends Thread {
	
	private InetAddress address;		// machine running max/msp
	private OSCPortOut sender;			// osc out
	private OSCMessage msg;			// osc packet
	private Object args[];				// osc content
	private String ip;
	private int soundID;				// incrementing sound ID
	public boolean audioEnabled = true;		// turns audio on/off
	public int gain = 1;				// default volume level
	public int clamp = 1;

	public SoundManager(String ip, int port){
		try{
			this.ip = ip;
			address = InetAddress.getByName(ip);		// a bad address will throw traxess parsing errors when using send!
			sender = new OSCPortOut(address, port);
		} catch (SocketException e){
			System.err.println(e);
		} catch (UnknownHostException e){
			System.err.println(e);
		}
		soundID = 0;
	}
	
/*	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
		System.out.println(filename);
		if(!filename.equals("none")){
			System.out.println(filename);
			soundID++;
			int[] speaker = getNearestSpeaker(x,y);
			send("simple instance"+soundID+" "+filename+" "+speaker[0]+" "+speaker[1]+" 0 "+gain+" "+comment);
		}
	}*/
	
	public void playSimpleSound(String filename, int c, float gain, String comment){ // c is channel number
		// this version takes a channel as argument and sends the appropriate coords to Marc Nimoy's max patch
		//System.out.println(filename);
		if(!filename.equals("none")){
			//System.out.println(filename);
			soundID++;
			int[] channelCoords = lookupCoordinates(c);
			//System.out.println("Played sound on channel " +c);
			send("simple instance"+soundID+" "+filename+" "+channelCoords[0]+" "+channelCoords[1]+" 0 "+gain+" "+comment);
		}
	}
	
	public void globalSound(int soundIDToStart, String filename, boolean loop, float gain, int duration, String comment) {
		if(!filename.equals("none")){
			//System.out.println(filename);
			// duration not used, no looping
			// send simple instanceID soundfilename.wav 0 0 1 1
			
			send("global instance"+soundIDToStart+" "+filename+" "+gain+" "+comment);
			//send("global instance"+soundIDToStart+" "+filename+" "+gain);

			//send("simple instance"+soundIDToStart+" "+soundFile+" "+0+" "+0+" "+0+" "+gain+" "+comment);
		}
	}
	
	private int[] lookupCoordinates(int c){
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
