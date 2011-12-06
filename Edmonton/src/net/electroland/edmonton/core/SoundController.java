package net.electroland.edmonton.core;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;

import net.electroland.ea.AnimationManager;
import net.electroland.scSoundControl.SCSoundControl;
import net.electroland.scSoundControl.SCSoundControlNotifiable;
import net.electroland.scSoundControl.SoundNode;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SoundController implements SCSoundControlNotifiable {

	private Hashtable<String,Object> context;
	private ElectrolandProperties props;
	private AnimationManager anim;
	private String soundFilePath;
	private boolean bypass;

	private SCSoundControl ss;
	private boolean serverIsLive;
	private Hashtable<String, Integer> soundFiles;
	private int soundID; // incrementing sound ID

	public boolean audioEnabled = true;

	static Logger logger = Logger.getLogger(SoundController.class);


	public SoundController(Hashtable<String,Object> context){

		this.context = context;
		props = (ElectrolandProperties) context.get("props");
		//anim = (AnimationManager) context.get("anim");

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

			// don't do this here, wait for server to report as live
			// parseSoundFiles();
		}


		logger.info("SoundController: started up with path " + soundFilePath + " and bypass=" + bypass);
	}

	public void parseSoundFiles(){
		/*
		 * Iterate across all animation Clips and look for soundfiles
		 * make a list and then buffer.
		 */


		// load anim props
		ElectrolandProperties p = new ElectrolandProperties(context.get("animpropsfile").toString());
		// rip clips for $soundfiles
		Map<String, ParameterMap> clipParams = p.getObjects("clip");
		for (String s : clipParams.keySet()){
			ParameterMap params = clipParams.get(s);

			String clipFileParams = params.getOptional("soundfiles");			
			if (clipFileParams != null){
				String[] fileList = clipFileParams.split(",");
				//logger.info("SOUNDMANAGER - clip soundFiles: " + fileList);
				for(int i=0; i<fileList.length; i++){
					if(!soundFiles.containsKey(soundFilePath+fileList[i])){ // have to include full path because that is what sc returns for check later
						//logger.info("SoundFiles did not contain key " + soundFilePath+fileList[i]);
						//load the buffer
						loadBuffer(fileList[i]);
						// put a ref to the buffer in soundFiles to mark it as loaded later
						soundFiles.put(soundFilePath+fileList[i], -1);	// -1 default unassigned value
					}
				}
			}
		}
		
		// rip sound.global for $soundfiles
		Map<String, ParameterMap> soundParams = p.getObjects("sound");
		for (String s : soundParams.keySet()){
			ParameterMap params = soundParams.get(s);

			String globalFileParams = params.getOptional("soundfiles");			
			if (globalFileParams != null){
				String[] fileList = globalFileParams.split(",");
				for(int i=0; i<fileList.length; i++){
					if(!soundFiles.containsKey(soundFilePath+fileList[i])){ // have to include full path because that is what sc returns for check later
						//load the buffer
						loadBuffer(fileList[i]);
						// put a ref to the buffer in soundFiles to mark it as loaded later
						soundFiles.put(soundFilePath+fileList[i], -1);	// -1 default unassigned value
					}
				}
			}
		}



		// debug - list the soundFiles
		logger.info("SoundController: List of ripped soundfiles"); 
		for (String s : soundFiles.keySet()){
			logger.info("\tkey " + s + " = " + soundFiles.get(s)); 
		}


	}

	private int getMappedChannelID (int bayNum) {
		//TO DO create a mapper specific to MOTU output here
		return 0;
	}

	public void loadBuffer(String soundFile){

		ss.readBuf(soundFilePath+soundFile);

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

	public void playTestSound(String filename){
		if (!bypass) {

			soundID++;
			if(!filename.equals("none") && serverIsLive){
				int channel = 0; //test
				SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, 1.0f, 1.0f);
				logger.info("SoundController: Played test sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
			}
		}
	}
	
	public void playSimpleSound(String filename, int x, int y, float gain, String comment){
		if (!bypass) {


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
				logger.info("SoundController: Played test sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
			}
		}
	}
	
	public void playSingleBaySound(String filename, int x, float masterGain, String comment){
		if (!bypass) {

			soundID++;
			if(!filename.equals("none") && serverIsLive){
				int channel = 0;
				
				SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, masterGain, 1.0f);
				logger.info("SoundController: Played test sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
			}
		}
	}

	public void globalSound(String soundFile, boolean loop, float gain, String comment){
		if (!bypass) {
			if(!soundFile.equals("none") && serverIsLive){
				// whoah, hacky.  let's fixt this
				int[] channels = new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
				float[] amplitudes = new float[]{gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain};
				SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+soundFile), false, channels, amplitudes, 1.0f);
				logger.info("SoundController: Played global sound file "+soundFilePath+soundFile+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
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
		logger.info("SoundController: Loaded buffer " + id + ", " + filename);
		
		if(soundFiles.containsKey(filename)){
			soundFiles.put(filename, id);	// update the sound file reference to the buffer ID
		}
	}

	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		//for now disable this for realtime loading
		parseSoundFiles();
	}

	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU, int numSynths) {
		// keep track of server load		
	}

	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
	}


}

