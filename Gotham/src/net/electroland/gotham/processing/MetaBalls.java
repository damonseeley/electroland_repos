package net.electroland.gotham.processing;

import processing.core.PVector;

/**
 * Adapted from http://www.openprocessing.org/sketch/13250
 */

public class MetaBalls extends GothamPApplet {

    private static final long serialVersionUID = 7554567212859904536L;

    final int NUM_BALLS         = 6;
    final float FRICTION        = 0.999f;
    final float MOUSE_REPEL     = 60f;
    final float MAX_VEL         = 5;
    final float COHESION_WEIGHT = 0.05f;
    final float THRESH_1        = 5;
    final float THRESH_2        = 5.5f;
    final float THRESH_3        = 6;
    final float THRESH_4        = 7;

    // TODO: should cycle hue and saturation from an image file
    private int hue               = 0;
    private int saturation        = 100;

    // might be nice to clean these up by putting them in a Metaball object
    private float[] mbRadius    = new float[NUM_BALLS];
    private PVector[] mbPos     = new PVector[NUM_BALLS];
    private PVector[] mbVel     = new PVector[NUM_BALLS];

    private PVector center      = new PVector(0,0); // need to understand this.

    @Override
    public void setup(){
        colorMode(HSB,360,100,100);
        for(int i=0; i< NUM_BALLS; i++) {
          mbPos[i] = new PVector(random(0,this.getSyncArea().width),random(0,getSyncArea().height));
          mbVel[i] = new PVector(random(-1,1),random(-1,1));
          mbRadius[i] = random(90,140);
        }
    }

    @Override
    public void drawELUContent() {

        center.set(0,0,0);
        //update meta ball positions
        for(int i=0;i<NUM_BALLS;i++) {
          // we'll replace mouse with vision
//          if(mousePressed) {
//            PVector m = new PVector();
//            m.set(mouseX,mouseY,0);
//            float mDistance = PVector.dist(mbPos[i],m);
//            PVector repel = PVector.sub(mbPos[i],m);
//            repel.normalize();
//            repel.mult(MOUSE_REPEL/(mDistance*mDistance));
//            mbVel[i].add(repel);
//          }

          center.add(mbPos[i]);
          mbVel[i].mult(FRICTION);
          mbVel[i].limit(MAX_VEL);
        }

        center.div(NUM_BALLS);

        for(int i=0;i<NUM_BALLS;i++) {

          // gravity to center
          PVector c = PVector.sub(center,mbPos[i]);
          c.normalize();
          c.mult(COHESION_WEIGHT);
          mbVel[i].add(c);

          mbPos[i].add(mbVel[i]);

          // wall bouncing
          if(mbPos[i].x > this.getSyncArea().width) {
            mbPos[i].x = this.getSyncArea().width;
            mbVel[i].x *= -1.0;
          }
          if(mbPos[i].x < 0) {
            mbPos[i].x = 0;
            mbVel[i].x *= -1.0;
          }
          if(mbPos[i].y > this.getSyncArea().height) {
            mbPos[i].y = this.getSyncArea().height;
            mbVel[i].y *= -1.0;
          }
          if(mbPos[i].y < 0) {
            mbPos[i].y = 0;
            mbVel[i].y *= -1.0;
          }
        }

        // render
        background(0);
        hue++; // cycle hue
        float sum;
        for(int i=0; i<width; i++) {
          for(int j=0; j<height; j++) {
            sum = 0;
            for(int m=0; m<NUM_BALLS; m++) {
              sum += mbRadius[m] / sqrt(sq(i-mbPos[m].x) + sq(j-mbPos[m].y));
            }
            set(i,j,color(hue%360, saturation, (sum*sum*sum)/3));
          }
        }
    }
}