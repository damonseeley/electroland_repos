package net.electroland.minimsoundtest.core;

import processing.core.*;
import ddf.minim.*;
import ddf.minim.signals.*;
import ddf.minim.analysis.*;
import ddf.minim.effects.*;

public class BasicTest extends PApplet {

    Minim minim;   
    AudioPlayer player;

    public void setup() {

        size(512, 200);

        minim = new Minim(this);
        
     // loadFile will look in all the same places as loadImage does.
        // this means you can find files that are in the data folder and the 
        // sketch folder. you can also pass an absolute path, or a URL.
        player = minim.loadFile("01_Aspera.mp3");
        
        // play the file
        player.play();
    }
    
    public void draw()
    {
      background(0);
      stroke(255);
      
      // draw the waveforms
      // the values returned by left.get() and right.get() will be between -1 and 1,
      // so we need to scale them up to see the waveform
      // note that if the file is MONO, left.get() and right.get() will return the same value
      for(int i = 0; i < player.bufferSize() - 1; i++)
      {
        float x1 = map( i, 0, player.bufferSize(), 0, width );
        float x2 = map( i+1, 0, player.bufferSize(), 0, width );
        line( x1, 50 + player.left.get(i)*50, x2, 50 + player.left.get(i+1)*50 );
        line( x1, 150 + player.right.get(i)*50, x2, 150 + player.right.get(i+1)*50 );
      }
    }



    /**
     * @param args
     */
     public static void main(String[] args) {
         // TODO Auto-generated method stub
         BasicTest bt = new BasicTest();
     }

}