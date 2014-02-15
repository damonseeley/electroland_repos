package net.electroland.enteractive.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import net.electroland.scSoundControl.SCSoundControl;
import net.electroland.scSoundControl.SCSoundControlNotifiable;
import net.electroland.scSoundControl.SoundNode;

/**
 * SoundManager based on LAFM, uses Super Collider.
 * @author asiegel
 */

//change test

public class SoundManagerSCSC implements SCSoundControlNotifiable {
	
	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private List<Speaker> speakers;
	private String absolutePath;
	public Properties soundProps;
	
	public SoundManagerSCSC(){
		
		try{
			soundProps = new Properties();
			soundProps.load(new FileInputStream(new File("depends//sounds.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		serverIsLive = false;
		soundFiles = new Hashtable<String, Integer>();
		speakers = new ArrayList<Speaker>();
		absolutePath = soundProps.getProperty("path");
		ss = new SCSoundControl(this);
		ss.init();
		ss.showDebugOutput(false);
		ss.set_serverResponseTimeout(5000);
		parseSpeakers();
	}
	
	public SoundNode createLoopingSound(String filename, float x, float y, float width, float height){
		if(!filename.equals("none") && serverIsLive){
			Calendar cal = new GregorianCalendar();
			int currentHour = 0;
			if(cal.get(Calendar.HOUR) == 0){
				currentHour = 12;
			} else {
				currentHour = cal.get(Calendar.HOUR);
			}
			if(cal.get(Calendar.AM_PM) > 0){
				currentHour += 12;
			}
			if(currentHour >= Integer.parseInt(soundProps.getProperty("soundsOn")) && currentHour < Integer.parseInt(soundProps.getProperty("soundsOff"))){
				float[] amplitudes = getAmplitudes(x, y, width, height);
				return ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), true, amplitudes, 1.0f);	
			}
			
		}
		return null;
	}
	
	public SoundNode createMonoSound(String filename, float x, float y, float width, float height){
		if(!filename.equals("none") && serverIsLive){
			Calendar cal = new GregorianCalendar();
			int currentHour = 0;
			if(cal.get(Calendar.HOUR) == 0){
				currentHour = 12;
			} else {
				currentHour = cal.get(Calendar.HOUR);
			}
			if(cal.get(Calendar.AM_PM) > 0){
				currentHour += 12;
			}
			if(currentHour >= Integer.parseInt(soundProps.getProperty("soundsOn")) && currentHour < Integer.parseInt(soundProps.getProperty("soundsOff"))){
				float[] amplitudes = getAmplitudes(x, y, width, height);
				//System.out.println(amplitudes.length);
				for(int i=0; i<amplitudes.length; i++){
					//System.out.print(amplitudes[i]+" ");
				}
				return ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), false, amplitudes, 1.0f);
			}
		}
		return null;
	}
	
	public void loadBuffer(String soundFile){
		ss.readBuf(soundFile);
	}
	
	public float[] getAmplitudes(float x, float width){
		// one-dimensional, ie: stereo panning of amplitude values
		float[] amplitudes = new float[speakers.size()];
		Iterator<Speaker> iter = speakers.iterator();
		int i = 0;
		while(iter.hasNext()){
			Speaker s = iter.next();
			amplitudes[i] = s.getAmplitude(x/width);	// normalize location
			i++;
		}
		return amplitudes;
	}
	
	public float[] getAmplitudes(float x, float y, float width, float height){
		float[] amplitudes = new float[speakers.size()];
		Iterator<Speaker> iter = speakers.iterator();
		int i = 0;
		while(iter.hasNext()){
			Speaker s = iter.next();
			amplitudes[i] = s.getAmplitude(x/width, y/height);	// normalize location
			i++;
		}
		return amplitudes;
	}
	
	public void parseSpeakers(){
		Iterator<Map.Entry<Object,Object>> iter = soundProps.entrySet().iterator();
		while(iter.hasNext()){
			Map.Entry<Object,Object> entry = iter.next();
			if(entry.getKey().toString().startsWith("speaker")){
				String[] loc = entry.getValue().toString().split(",");
				speakers.add(new Speaker(Float.parseFloat(loc[0]), Float.parseFloat(loc[1])));
			}
		}
	}
	
	public void parseSoundFiles(Properties sysProps){
		Iterator<Object> iter = sysProps.values().iterator();
		while(iter.hasNext()){
			String prop = (String)iter.next();
			String[] proplist = prop.split(",");
			for(int i=0; i<proplist.length; i++){
				if(proplist[i].endsWith(".wav")){
					if(!soundFiles.containsKey(proplist[i])){
						//System.out.println(proplist[i]);	// print out sound file to make sure it's working
						soundFiles.put(absolutePath+proplist[i], -1);	// -1 default unassigned value
						loadBuffer(absolutePath+proplist[i]);
					}
				}
			}
		}
	}

	public void receiveNotification_BufferLoaded(int id, String filename) {
		System.out.println("Loaded buffer " + id + ", " + filename);
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}		
	}

	public void receiveNotification_ServerRunning() {
		//if(!serverIsLive){
			serverIsLive = true;
			parseSoundFiles(soundProps);
		//}
	}

	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU) {
		// TODO Keep track of server load
	}

	public void receiveNotification_ServerStopped() {
		//serverIsLive = false;
	}
	
	public void receiveNotification_ServerStatus(float averageCPU,
			float peakCPU, int numSynths) {
		// TODO Auto-generated method stub
		
	}
	
	public void killAll(){
		ss.init();
	}
	
	public void shutdown(){
		ss.shutdown();
	}
	
	
	
	
	public class Speaker{
		
		public float x, y;
		
		public Speaker(float x, float y){
			this.x = x;
			this.y = y;
		}
		
		public float getAmplitude(float targetx){
			return 1 - Math.abs(x - targetx);
		}
		
		public float getAmplitude(float targetx, float targety){
			// takes normalized location values and creates amplitude
			// from inverted value of hypotenuse distance.
			float xdiff = x - targetx;
			float ydiff = y - targety;
			return 1 - (float)Math.sqrt((xdiff*xdiff) + (ydiff*ydiff));
		}
	}

}
