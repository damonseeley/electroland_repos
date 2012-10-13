package net.electroland.gotham.processing;

import processing.core.PVector;

/**
 * Adapted from http://www.openprocessing.org/sketch/13250
 */
public class MetaBalls extends GothamPApplet {

    private static final long serialVersionUID = 7554567212859904536L;

    final int   NUM_BALLS       = 6;
    final int   MIN_RADIUS      = 90;    // initial ball radius min
    final int   MAX_RADIUS      = 140;   // initial ball radius max

    final float FRICTION        = 1.5f;  // higher = less
    final float MAX_VEL         = 15;    // higher = faster
    final float COHESION_WEIGHT = 1.0f;  // higher = more

    // TODO: should cycle hue and saturation from an image file
    private int hue             = 0;
    private int saturation      = 100;

    // might be nice to clean these up by putting them in a Collection of Metaballs.
    private float[]   mbRadius  = new float[NUM_BALLS];
    private PVector[] mbPos     = new PVector[NUM_BALLS];
    private PVector[] mbVel     = new PVector[NUM_BALLS];

    @Override
    public void setup(){
        colorMode(HSB, 360, 100, 100);
        for(int i = 0; i < NUM_BALLS; i++) {
          mbPos[i]    = new PVector(random(0, this.getSyncArea().width),random(0, getSyncArea().height));
          mbVel[i]    = new PVector(random(-1, 1),random(-1, 1));
          mbRadius[i] = random(MIN_RADIUS, MAX_RADIUS);
        }
    }

    @Override
    public void drawELUContent() {

        PVector center = new PVector(0, 0, 0);

        //update meta ball positions
        for(int i=0; i < NUM_BALLS; i++) {
            center.add(mbPos[i]);
            mbVel[i].mult(FRICTION);
            mbVel[i].limit(MAX_VEL);
        }

        center.div(NUM_BALLS);

        for(int i = 0; i < NUM_BALLS; i++) {

            // gravity to center
            PVector c = PVector.sub(center, mbPos[i]);
            c.normalize();
            c.mult(COHESION_WEIGHT);
            mbVel[i].add(c);
            mbPos[i].add(mbVel[i]);

              // simple bounce when beyond bounds.  only accomplishes 90 or 180
              // degree turns.
            if(mbPos[i].x > this.getSyncArea().width) {
                mbPos[i].x = this.getSyncArea().width;
                mbVel[i].x = -mbVel[i].x;
            }
            if(mbPos[i].x < 0) {
                mbPos[i].x = 0;
                mbVel[i].x = -mbVel[i].x;
            }
            if(mbPos[i].y > this.getSyncArea().height) {
                mbPos[i].y = this.getSyncArea().height;
                mbVel[i].y = -mbVel[i].y;
            }
            if(mbPos[i].y < 0) {
                mbPos[i].y = 0;
                mbVel[i].y = -mbVel[i].y;
            }
        }

        // render
        background(0);
        hue++; // cycle hue

        // given that computation goes up with square of dimensions, it would
        // make sense to do something like determine the boundaries of the
        // detectors and only render that range.
        
        for(int i = 0; i < this.getSyncArea().width; i++) {
            for(int j = 0; j < this.getSyncArea().height; j++) {
                float sum = 0;
                for(int m = 0; m < NUM_BALLS; m++) {
                    sum += mbRadius[m] / sqrt(sq(i - mbPos[m].x) + sq(j - mbPos[m].y));
                }
                set(i, j, color(hue % 360, saturation, (sum * sum * sum) / 3));
            }
        }
    }
}