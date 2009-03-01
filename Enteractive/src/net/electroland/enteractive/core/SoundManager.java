package net.electroland.enteractive.core;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Properties;

import net.electroland.scSoundControl.SCSoundControl;
import net.electroland.scSoundControl.SCSoundControlNotifiable;
import net.electroland.scSoundControl.SoundNode;

/**
 * SoundManager based on LAFM, uses Super Collider.
 * @author asiegel
 */

public class SoundManager implements SCSoundControlNotifiable {
	
	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private String absolutePath;
	private Properties systemProps;
	
	public SoundManager(int numOutputChannels, int numInputChannels, Properties systemProps){
		this.systemProps = systemProps;
		serverIsLive = false;
		soundFiles = new Hashtable<String, Integer>();
		absolutePath = "D:\\Programming\\Java\\Enteractive\\soundfiles\\";	// should come from systemProps
		ss = new SCSoundControl(this);
		ss.init();
		ss.showDebugOutput(true);
	}
	
	public SoundNode createMonoSound(String filename, float gain){
		if(!filename.equals("none") && serverIsLive){
			// need to look up coordinates here to gauge amplitudes
			//return ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), false, amplitudes, 1.0f);
		}
		return null;
	}
	
	public void loadBuffer(String soundFile){
		ss.readBuf(soundFile);
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
		parseSoundFiles(systemProps);		
	}

	@Override
	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU) {
		// TODO Keep track of server load
	}

	@Override
	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
	}

}
