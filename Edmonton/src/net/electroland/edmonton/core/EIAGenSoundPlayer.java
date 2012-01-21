package net.electroland.edmonton.core;

import java.awt.Color;
import java.util.ArrayList;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;

import org.apache.log4j.Logger;

public class EIAGenSoundPlayer {

    private SoundController sc;
    private double x;
    
    private ArrayList<String> bondi = new ArrayList<String>();
    private ArrayList<String> elevation = new ArrayList<String>();
    private ArrayList<String> kotu = new ArrayList<String>();
    private ArrayList<String> marimbaH = new ArrayList<String>();
    private ArrayList<String> marimbaL = new ArrayList<String>();

    static Logger logger = Logger.getLogger(EIAGenSoundPlayer.class);

    public EIAGenSoundPlayer(SoundController sc)
    {
        
        this.sc = sc;
        
        for (int i=1;i<=5;i++){
            bondi.add("bondi_0"+i+".wav");
        }
        for (int i=1;i<=5;i++){
            elevation.add("elevation_0"+i+".wav");
        }
        for (int i=1;i<=5;i++){
            kotu.add("kotu_0"+i+".wav");
        }
        for (int i=1;i<=6;i++){
            marimbaH.add("marimba_high_0"+i+".wav");
        }
        for (int i=1;i<=6;i++){
            marimbaL.add("marimba_mid_0"+i+".wav");
        }
        
        //logger.info("GENSOUNDS: " + bondi + elevation + kotu + marimbaH + marimbaL);
        
    }


    public void playNextGen(double x) {
        
        this.x = x;
        
        double chance = Math.random()*100;
        
        if (chance < 30){
            playSound(bondi);
        } else if (chance < 65){
            //playSound(kotu);
            playSound(marimbaL);
        } else if (chance < 100){
            playSound(marimbaH);
        }
        
        if (chance > 90){
            playSound(elevation);
        }

      
    }
    
    private void playSound(ArrayList<String> soundList) {
        //logger.info("playNextGen @ " + x + " " + soundList);
        
        String soundtoplay = (String) soundList.get((int) (Math.random()*soundList.size()));
        //logger.info("GENSOUND: playing " + soundtoplay );
        if (sc != null) {
            //sc.playSingleChannel(soundtoplay, x, 1.0f);
            sc.playGlobal(soundtoplay, false, 1.0f);
        }
        
        
    }
    
    

}