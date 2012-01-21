package net.electroland.edmonton.core;

import java.util.ArrayList;
import java.util.Hashtable;
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

    final static int[] multichannels = {2,3,4,5,6,7,8,9,14,15,16};
    final static int[] stereochannels = {0,1};

    private Hashtable<String,Object> context;
    private ElectrolandProperties props;
    private AnimationManager anim;
    private String soundFilePath;
    private boolean bypass;

    private SCSoundControl ss;
    private boolean serverIsLive;
    private Hashtable<String, Integer> soundFiles;
    private Hashtable<Integer, Speaker> speakers;
    private Hashtable<Integer, SoundNode> soundNodes;
    private int soundID; // incrementing sound ID

    public boolean audioEnabled = true;

    private boolean debug;
    private boolean stereoOnly;

    static Logger logger = Logger.getLogger(SoundController.class);

    public SoundController(Hashtable<String,Object> context){

        this.context = context;
        props = (ElectrolandProperties) context.get("props");
        //anim = (AnimationManager) context.get("anim");

        soundID = 0;
        soundNodes = new Hashtable<Integer,SoundNode>();

        try {
            this.soundFilePath = props.getOptional("settings", "sound", "filePath");
            this.bypass = Boolean.parseBoolean(props.getOptional("settings", "sound", "soundBypass"));
            this.debug = Boolean.parseBoolean(props.getOptional("settings", "sound", "debug"));
            this.stereoOnly = Boolean.parseBoolean(props.getOptional("settings", "sound", "stereoOnly"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            soundFilePath = "/depends/";
            bypass = false;
            debug = false;
            stereoOnly = false;
        }


        // rip the speaker ID and locations
        speakers = new Hashtable<Integer, Speaker>(); //fill this with soundfilenames and mark them as loaded or not
        Map<String, ParameterMap> speakerParams = props.getObjects("speaker");
        for (String s : speakerParams.keySet()){
            ParameterMap sParams = speakerParams.get(s);

            int bay = sParams.getRequiredInt("bay");
            double x = sParams.getRequiredDouble("x");
            double y = sParams.getRequiredDouble("y");
            int channel = sParams.getRequiredInt("ch");

            Speaker sp = new Speaker(bay,x,y,channel);
            speakers.put(bay, sp);
        }

        logger.info("SoundController: List of speakers"); 
        for (int sp : speakers.keySet()){
            logger.info("\tbay" + sp + " x:" + speakers.get(sp).x + " y:" + speakers.get(sp).y + " ch:" + speakers.get(sp).channel); 
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


        boolean testme = false;
        if (testme) {
            logger.info("GetClosestBay for 610.0 (bay 1): " + getClosestBay(610.0));
            logger.info("GetClosestBay for 560.0 (bay 2): " + getClosestBay(560.0));
            logger.info("GetClosestBay for 501.0 (bay 3): " + getClosestBay(501.0));
            logger.info("GetClosestBay for 453.0 (bay 4): " + getClosestBay(453.0));
            logger.info("GetClosestBay for 405.0 (bay 5): " + getClosestBay(405.0));
            logger.info("GetClosestBay for 361.0 (bay 6): " + getClosestBay(361.0));
            logger.info("GetClosestBay for 216.0 (bay 7): " + getClosestBay(216.0));
            logger.info("GetClosestBay for 168.0 (bay 8): " + getClosestBay(168.0));
            logger.info("GetClosestBay for 128.0 (bay 9): " + getClosestBay(102.0));
            logger.info("GetClosestBay for 80.0 (bay 10): " + getClosestBay(98.0));
            logger.info("GetClosestBay for 39.0 (bay 11): " + getClosestBay(50.0));
        }

        if (debug) {
            logger.info("SoundController: started up with path " + soundFilePath + " and bypass=" + bypass);
        }
    }

    public void parseSoundFiles(){
        /*
         * Iterate across all animation Clips and look for soundfiles
         * make a list and then buffer.
         */


        // load anim props
        ElectrolandProperties p = new ElectrolandProperties(context.get("animpropsfile").toString());
        // rip clips for $soundfiles
        Map<String, ParameterMap> clipParams = p.getObjects("content");
        for (String s : clipParams.keySet()){
            ParameterMap params = clipParams.get(s);

            String clipFileParams = params.getOptional("soundfiles");			
            if (clipFileParams != null){
                String[] fileList = clipFileParams.split(",");
                //logger.info("SOUNDMANAGER - clip soundFiles: " + fileList);
                for(int i=0; i<fileList.length; i++){
                    if(!soundFiles.containsKey(soundFilePath+fileList[i])){ // have to include full path because that is what sc returns for check later
                        //logger.info("SoundFiles did not contain key " + soundFilePath+fileList[i]);
                        //load the buffer, no, do it later now
                        //loadBuffer(fileList[i]);
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
                        //load the buffer, no, do it later now
                        //loadBuffer(fileList[i]);
                        // put a ref to the buffer in soundFiles to mark it as loaded later
                        soundFiles.put(soundFilePath+fileList[i], -1);	// -1 default unassigned value
                    }
                }
            }
        }


        //load sequencer props to get soundfile references
        ElectrolandProperties ps = new ElectrolandProperties(context.get("seqpropsfile").toString());
        for (String showName : ps.getObjectNames("show"))
        {
            for (String name : ps.getObjectNames(showName))
            {
                if (name.startsWith("soundcue.")){
                    ParameterMap params = ps.getParams(showName, name);
                    //String soundCueId = name.substring(9, name.length() - 9);
                    String soundFileName = params.getRequired("soundfilename");
                    //logger.info("SoundController: sequence soundfilename ripper found: " + soundFileName);
                    soundFiles.put(soundFilePath+soundFileName, -1);	// -1 default unassigned value
                }
            }
        }



        loadAllBuffers();

        // debug - list the soundFiles
        if (debug) {
            logger.info("SoundController: List of ripped soundfiles, total=" + soundFiles.size()); 
            for (String s : soundFiles.keySet()){
                logger.info("\tkey " + s + " = " + soundFiles.get(s)); 
            }
        }

    }


    public void loadAllBuffers() {
        for (String s : soundFiles.keySet()){
            //logger.info("SoundController: loadbuffer: " + s);
            ss.readBuf(s);
        }
    }


    public void loadBuffer(String soundFile){
        ss.readBuf(soundFilePath+soundFile);
    }





    /*
     * Utils for mapping output to bays and channels
     */

    private int getClosestBayChannel (double x){
        return getChID(getClosestBay(x));
    }

    /**
     * Return the correct device channel value for a given bay number
     * ... unless stereoOnly is true in which case return 0 (Left)
     * @param bayNum
     * @return
     */
    private int getChID (int bayNum) {
        //TO DO create a mapping specific to MOTU output here
        if (stereoOnly){
            return 1; //for this project will map to computer channel 1 (Left)
        }
        else{
            return speakers.get(bayNum).channel;
        }
    }

    private int getClosestBay (double x){
        Speaker closest = speakers.get(1);
        for (int sp : speakers.keySet()){
            //logger.info("Closest = " + closest.bay + " and current calc = " + Math.abs(x - speakers.get(sp).x))
            if (Math.abs(x - speakers.get(sp).x) < Math.abs(x - closest.x)) {
                closest = speakers.get(sp);
            }
        }
        return closest.bay;
    }


    
    public void fadeAll(int duration)
    {
        logger.info("fading audio out");
        // copy all live sounds and empty out the sound state
        synchronized (soundNodes){
            ArrayList<SoundNode> fadeNodes = new ArrayList<SoundNode>();
            fadeNodes.addAll(soundNodes.values());
            soundNodes.clear();
            if (stereoOnly){
                new AudioFadeThread(fadeNodes, stereochannels, this, 500).start();
            }else{
                new AudioFadeThread(fadeNodes, multichannels, this, 500).start();
            }
        }
    }

    public void kill(SoundNode n)
    {
        n.die();
        n.cleanup();
        soundNodes.remove(n);
    }

    /** 
     * play a test sound at 1.0f gain out of the first channel
     * @param filename
     */
    public void playTestSound(String filename){
        if (!bypass) {
            if(!filename.equals("none") && serverIsLive){
                int channel = 0; //test
                SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, 1.0f, 1.0f);
                if (debug) {
                    logger.info("SoundController: Played test sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
                }
                newSoundNode(sn);
            }
        }
    }

    /**
     * Plays a sound only to the speaker nearest to the x value provided
     * Does not provide a SoundNode reference for update
     * @param filename (String) the sound file to play (without path)
     * @param x (double) location
     * @param gain (float)
     */
    public void playSingleChannel(String filename, double x, float gain){
        if (!bypass) {
            if(!filename.equals("none") && serverIsLive){
                
                int channel = getClosestBayChannel(x);
                
                if (stereoOnly){
                    int[] channels = stereochannels;
                    float[] amplitudes = new float[]{gain,gain};
                    SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+filename), false, channels, amplitudes, 1.0f);
                    if (debug) {
                        logger.info("SoundController: Played global sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
                    }
                    newSoundNode(sn);
                } else {
                    // whoah, hacky.  let's fix this
                    // what we're doing here is hard coding Edmonton channel IDs and gain values to correspond with MOTU hardware
                    int[] channels = new int[]{channel};
                    float[] amplitudes = new float[]{gain};
                    SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+filename), false, channels, amplitudes, 1.0f);
                    if (debug) {
                        logger.info("SoundController: Played global sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
                    }
                    newSoundNode(sn);
                }
                
                
                /*
                int channel = getClosestBayChannel(x);
                if (debug) {
                    logger.info("SoundController: Attempting to play sound file: " + soundFiles.get(soundFilePath+filename));
                }
                SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, gain, 1.0f);
                if (debug) {
                    logger.info("SoundController: Played Single Bay "+soundFilePath+filename+ " on channel: " + channel + " with gain: "+ gain);
                }
                newSoundNode(sn);
                */
            }
        }
    }

    public void playLocal(String filename, double x, float gain){
        if (!bypass) {
            //temp
            playSingleChannel(filename, x, gain);

            //TO DO make this work
            /*
			soundID++;
			if(!filename.equals("none") && serverIsLive){
				channel = ??
				SoundNode sn = ss.createSoundNodeOnSingleChannel(soundFiles.get(soundFilePath+filename), false, channel, masterGain, 1.0f);
				if (debug) {
				logger.info("SoundController: Played test sound file "+soundFilePath+filename+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
				}
			}
             */
        }
    }

    public void playGlobal(String soundFile, boolean loop, float gain){
        if (!bypass) {
            if(!soundFile.equals("none") && serverIsLive){
                if (stereoOnly){
                    int[] channels = new int[]{0,1};
                    float[] amplitudes = new float[]{gain,gain};
                    SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+soundFile), false, channels, amplitudes, 1.0f);
                    if (debug) {
                        logger.info("SoundController: Played global sound file "+soundFilePath+soundFile+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
                    }
                    newSoundNode(sn);
                } else {
                    // whoah, hacky.  let's fix this
                    // what we're doing here is hard coding Edmonton channel IDs and gain values to correspond with MOTU hardware
                    int[] channels = multichannels;
                    float[] amplitudes = new float[]{gain,gain,gain,gain,gain,gain,gain,gain,gain,gain,gain};
                    SoundNode sn = ss.createMonoSoundNode(soundFiles.get(soundFilePath+soundFile), false, channels, amplitudes, 1.0f);
                    if (debug) {
                        logger.info("SoundController: Played global sound file "+soundFilePath+soundFile+ " and got back node with bus " + sn.get_busID()+ " and group " + sn.getGroup());
                    }
                    newSoundNode(sn);
                }
            }
        }
    }

    public void newSoundNode(SoundNode sn){	// no longer in use, but referenced by all animations

        if (sn != null){
            soundID++;
            soundNodes.put(soundID, sn);
            logger.info("SoundController: soundNodes updated, size="+soundNodes.size());
        }

    }




    public void killAllSounds(){
        if (!bypass) {
            ss.freeAllBuffers();	
        }
    }

    public void shutdown(){
        if (!bypass) {
            if(ss != null){
                //logger.info("SoundController: shutting down SCSC: " + ss);
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
        logger.info("SoundController: Server is live");

        //for now disable this for realtime loading
        parseSoundFiles();  // we don't really need to re-rip props, just need to rebuf
        //loadAllBuffers();
    }

    public void receiveNotification_ServerStatus(float averageCPU, float peakCPU, int numSynths) {
        // keep track of server load		
    }

    public void receiveNotification_ServerStopped() {
        serverIsLive = false;
        logger.info("SoundController: Server stopped");
    }


}


class Speaker
{
    int bay;
    double x;
    double y;
    //real channel on output device
    int channel;

    public Speaker(int b, double x, double y, int ch){
        this.bay = b;
        this.x = x;
        this.y = y;
        this.channel = ch;
    }

}



