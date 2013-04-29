package net.electroland.minimsoundtest.core;

import processing.core.PApplet;
import ddf.minim.AudioPlayer;
import ddf.minim.Minim;

public class BasicTestNonApplet {

    public static void main(String args[]) {


//        BasicTestNonApplet btna = new BasicTestNonApplet();

        Minim minim = new Minim(new PApplet());        
        AudioPlayer p1 = minim.loadFile("01_Aspera.mp3");
        AudioPlayer p2 = minim.loadFile("PitchedDXCowbells.mp3");

        // play the file
        while(true){
            p2 = minim.loadFile("PitchedDXCowbells.mp3");
            p2.play();
                
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }
}