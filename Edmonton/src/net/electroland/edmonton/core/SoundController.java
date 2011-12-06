package net.electroland.edmonton.core;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;

import net.electroland.scSoundControl.SCSoundControl;
import net.electroland.scSoundControl.SCSoundControlNotifiable;
import net.electroland.scSoundControl.SoundNode;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;

import org.apache.log4j.Logger;

public class SoundController implements SCSoundControlNotifiable {

	private Hashtable context;
	private ElectrolandProperties props;
	private String soundFilePath;
	private boolean bypass;


	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private int soundID; // incrementing sound ID

	public boolean audioEnabled = true;

	static Logger logger = Logger.getLogger(SoundController.class);


	public SoundController(Hashtable context){

		this.context = context;
		props = (ElectrolandProperties) context.get("props");
		try {
			this.soundFilePath = props.getOptional("settings", "sound", "filePath");
			this.bypass = Boolean.parseBoolean(props.getOptional("settings", "sound", "soundBypass"));
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			soundFilePath = "/depends/";
			bypass = false;
		}

		if (!bypass) {
			serverIsLive = false;
			soundFiles = new Hashtable<String, Integer>(); //fill this with soundfilenames and mark them as loaded or not

			ss = new SCSoundControl(this);
			ss.init();
			ss.showDebugOutput(true);
			ss.set_serverResponseTimeout(5000);	
		}


		logger.info("SoundController started up with path " + soundFilePath + " and bypass=" + bypass);
	}

	public void parseSoundFiles(){
		/*
		 * Not sure we will need this if loading is done at play time, see if that works first
		 * change this to iterate across all animation Clips and look for soundfiles
		 * make a list and then buffer.
		 */

		if (!bypass) {

			
			
			
			
			
			/*
			Iterator<String> iter = systemProps.values().iterator();
			while(iter.hasNext()){
				String prop = (String)iter.next();
				String[] proplist = prop.split(",");
				for(int i=0; i<proplist.length; i++){
					if(proplist[i].endsWith(".wav")){
						if(!soundFiles.containsKey(proplist[i])){
							System.out.println(proplist[i]);	// print out sound file to make sure it's working
							soundFiles.put(soundFilePath+proplist[i], -1);	// -1 default unassigned value
							loadBuffer(proplist[i]);
						}
					}
				}
			}
			*/
			
			
		}
	}
	
	private int getMappedChannelID (int bayNum) {
		//TO DO create a mapper specific to MOTU output here
		return 0;
	}

	public void loadBuffer(String soundFile){
		if (!bypass) {
			ss.readBuf(soundFilePath+soundFile);
		}
	}


	/* this is Indy specific code I think
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
	 */


	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
		if (!bypass) {

			// This code attempts to load the file at playtime, we'll see if this works
			// note that the buffer ID is assigned later when scsc reports the buffer as loaded
			if (!soundFiles.containsKey(filename)) {
				loadBuffer(filename);
			}

			soundID++;
			if(!filename.equals("none") && serverIsLive){
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
				SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, gain, 1.0f);
				//float[] amplitudes = getAmplitudes(x, y, gain);
				//SoundNode sn = ss.createMonoSoundNode(soundFiles.get(absolutePath+filename), false, amplitudes, 1.0f);
				//if(sn == null){
				//	System.out.println(soundFiles.get(absolutePath+filename) + " returned null");	
				//}
			}
		}
	}
	
	public void globalSound(int soundIDToStart, String soundFile, boolean loop, float gain, int duration, String comment){
		if (!bypass) {
			if(!soundFile.equals("none") && serverIsLive){
				// whoah, hacky.  let's fixt this
				int[] channels = new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
				float[] amplitudes = new float[]{gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain};
				//float[] amplitudes = new float[]{gain,gain};
				//SoundNode sn = ss.createStereoSoundNodeWithLRMap(soundFiles.get(absolutePath+soundFile), false, new int[]{1, 0}, new int[]{0, 1}, 1.0f);
				SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+soundFile), false, channels, amplitudes, 1.0f);
			}
		}
	}

	public int newSoundID(){	// no longer in use, but referenced by all animations
		soundID++;
		return soundID;
	}
	
	


	public void killAllSounds(){
		if (!bypass) {
			ss.freeAllBuffers();	
		}
	}

	public void shutdown(){
		if (!bypass) {
			if(ss != null){
				ss.shutdown();
			}
		}
	}
	
	
	/**
	 * NOTIFICATIONS FROM SCSC
	 */

	public void receiveNotification_BufferLoaded(int id, String filename) {
		System.out.println("Loaded buffer " + id + ", " + filename);
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}
	}

	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		// for now disable this for realtime loading
		//parseSoundFiles();
	}

	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU, int numSynths) {
		// keep track of server load		
	}

	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
	}


}

