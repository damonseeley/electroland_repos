package net.electroland.norfolk.sound;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;
import net.electroland.utils.hours.OperatingHours;
import processing.core.PApplet;
import ddf.minim.AudioPlayer;
import ddf.minim.Minim;

public class SimpleSoundManager {

    private Map<String, SoundElement>playList;
    private Minim minim;
    private OperatingHours hours;

    public SimpleSoundManager(){
        minim = new Minim(new PApplet());
    }

    public SimpleSoundManager(OperatingHours hours){
        minim = new Minim(new PApplet());
        this.hours = hours;
    }

    public SimpleSoundManager(ElectrolandProperties props){
        minim = new Minim(new PApplet());
        load(props);
    }

    public SimpleSoundManager(PApplet applet, ElectrolandProperties props){
        minim = new Minim(applet);
        load(props);
    }

    public Collection<String> getPlayList(){
        return playList.keySet();
    }
    
    public void load(ElectrolandProperties props){
        Map<String,ParameterMap> soundProps = props.getObjects("sound");
        playList = new HashMap<String,SoundElement>();
        for (String id : soundProps.keySet()){
            SoundElement se = new SoundElement(soundProps.get(id).get("filename"),soundProps.get(id).getRequired("groupID"));
            playList.put(id, se);
        }
    }
    
    public void playSound(String soundName){
        if (hours.shouldBeOpenNow("sound")){
            AudioPlayer ap = minim.loadFile(playList.get(soundName).filename);
            new PlayThread(ap, ap.length() * 2).start();
        }
    }
}

class SoundElement {
	
	public String filename;
	public String groupID;
	
	public SoundElement(String fname, String gid) {
		filename = fname;
		groupID = gid;
		
	}
}

class PlayThread extends Thread{
   
   private AudioPlayer ap;
   private int millis;

   public PlayThread(AudioPlayer ap, int millis){
       this.ap = ap;
       this.millis = millis;
   }

   @Override
   public void run(){
       ap.play();
       try {
        Thread.sleep(millis);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
       ap.close();
   }
}