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

    private float[] mbRadius    = new float[NUM_BALLS];
    private float mDistance;
    private PVector[] mbPos     = new PVector[NUM_BALLS];
    private PVector[] mbVel     = new PVector[NUM_BALLS];
    private float sum           = 0.0f;
    private PVector m           = new PVector();
    private PVector c           = new PVector();
    private PVector repel       = new PVector();
    private PVector centre      = new PVector(0,0);

    boolean started = false;
    @Override
    public void drawELUContent() {

        if (!started){ // setup is flaking out.

            colorMode(HSB,360,100,100);
            int wSector = getSyncArea().width/NUM_BALLS;
            for(int i=0; i< NUM_BALLS; i++) {
              mbPos[i] = new PVector(random(0,wSector) + (wSector * i),random(0,getSyncArea().height));
              mbVel[i] = new PVector(random(-1,1),random(-1,1));
              mbRadius[i] = random(90,140);
            }
            started = true;
        }

        centre.set(0,0,0);
        //update meta ball positions
        for(int i=0;i<NUM_BALLS;i++) {
          // we'll replace mouse with vision
          if(mousePressed) {
            m.set(mouseX,mouseY,0);
            mDistance = PVector.dist(mbPos[i],m);
            repel = PVector.sub(mbPos[i],m);
            repel.normalize();
            repel.mult(MOUSE_REPEL/(mDistance*mDistance));
            mbVel[i].add(repel);
          }

          centre.add(mbPos[i]);
          mbVel[i].mult(FRICTION);
          mbVel[i].limit(MAX_VEL);
        }

        centre.div(NUM_BALLS);

        for(int i=0;i<NUM_BALLS;i++) {

          // gravity to center
          c = PVector.sub(centre,mbPos[i]);
          c.normalize();
          c.mult(COHESION_WEIGHT);
          mbVel[i].add(c);

          mbPos[i].add(mbVel[i]);

          // wall bouncing
          if(mbPos[i].x > width) {
            mbPos[i].x = width;
            mbVel[i].x *= -1.0;
          }
          if(mbPos[i].x < 0) {
            mbPos[i].x = 0;
            mbVel[i].x *= -1.0;
          }
          if(mbPos[i].y > height) {
            mbPos[i].y = height;
            mbVel[i].y *= -1.0;
          }
          if(mbPos[i].y < 0) {
            mbPos[i].y = 0;
            mbVel[i].y *= -1.0;
          }
        }

        // render
        background(0);
        for(int i=0; i<width; i++) {
          for(int j=0; j<height; j++) {
            sum = 0;
            for(int m=0; m<NUM_BALLS; m++) {
              sum += mbRadius[m] / sqrt(sq(i-mbPos[m].x) + sq(j-mbPos[m].y));
            }
            set(i,j,color(360,100,(sum*sum*sum)/3));
          }
        }
    }
}