package net.electroland.norfolk.sound;

import java.util.Collection;

import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;
import ddf.minim.Minim;

public class SimpleSoundManager {

    private Collection<String>playList;
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
        return playList;
    }

    public void load(ElectrolandProperties props){
        // TODO: load the song playlist here.
    }

    public void playSound(String soundName){
        minim.loadFile(soundName).play();
    }
}