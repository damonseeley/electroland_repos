package net.electroland.norfolk.sound;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;
import processing.core.PApplet;
import ddf.minim.Minim;

public class SimpleSoundManager {

    private Map<String, String>playList;
    private Minim minim;

    public SimpleSoundManager(){
        minim = new Minim(new PApplet());
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
        playList = new HashMap<String,String>();
        for (String id : soundProps.keySet()){
            playList.put(id, soundProps.get(id).get("filename"));
        }
    }

    public void playSound(String soundName){
        minim.loadFile(playList.get(soundName)).play();
    }
}