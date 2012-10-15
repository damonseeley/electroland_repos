package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;

import processing.core.PVector;

/**
 * Adapted from http://www.openprocessing.org/sketch/13250
 */
public class MetaBalls extends GothamPApplet {

    private static final long serialVersionUID = 7554567212859904536L;

    final int   COARSNESS       = 5;
    final int   NUM_BALLS       = 3;
    final int   MIN_RADIUS      = 90;    // initial ball radius min
    final int   MAX_RADIUS      = 200;   // initial ball radius max

    final float FRICTION        = 1.0f;  // higher = less
    final float MAX_VEL         = 15;    // higher = faster
    final float COHESION_WEIGHT = 0.05f;  // higher = more

    // TODO: should cycle hue and saturation from an image file
    private int hue             = 0;
    private int saturation      = 100;


    // might be nice to clean these up by putting them in a Collection of Metaballs.
    private List<Metaball> balls;

    @Override
    public void setup(){

        colorMode(HSB, 360, 100, 100);
        background(0);

        balls = new ArrayList<Metaball>();
        for(int i = 0; i < NUM_BALLS; i++) {
            balls.add(new Metaball(this.getSyncArea()));
        }
    }

    @Override
    public void drawELUContent() {

        PVector center = new PVector(0, 0, 0);

        //update meta ball positions
        for (Metaball ball : balls){
            center.add(ball.position);
            ball.velocity.mult(FRICTION);
            ball.velocity.limit(MAX_VEL);
        }

        center.div(NUM_BALLS);

        for (Metaball ball : balls){

            PVector c = PVector.sub(center, ball.position);
            c.normalize();
            c.mult(COHESION_WEIGHT);
            ball.velocity.add(c);
            ball.position.add(ball.velocity);

            ball.checkBounds();
        }

        // render
        hue++; // cheaply cycle hue

        // given that computation goes up with square of dimensions, it would
        // make sense to do something like determine the boundaries of the
        // detectors and only render that range.
        noStroke();
        for(int i = 0; i < this.getSyncArea().width; i+= COARSNESS) {
            for(int j = 0; j < this.getSyncArea().height; j+= COARSNESS) {
                float sum = 0;
                for (Metaball ball : balls){
                    // radius divided by distance from this pixel to the center of the ball.  meaning max = ball.radius 
                    ball.pixelImpact = ball.radius / sqrt(sq(i - ball.position.x) + sq(j - ball.position.y));
                    sum += ball.pixelImpact;
                }
                // TODO: need to calculate hue in here as well
                // hue should be the number above (being added to sum) as a coefficient blending the
                // current balls hue into the overall hue of the pixel
                //set(i, j, color(hue % 360, saturation, (sum * sum * sum) / 3));
                fill(hue % 360, saturation, (sum * sum * sum) / 3);
                this.rect(i, j, COARSNESS, COARSNESS);
            }
        }
    }

    class Metaball {

        float radius;
        PVector position;
        PVector velocity;
        int hue; // can't quite implement per ball hue yet.
        Dimension area;
        float pixelImpact;

        public Metaball(Dimension initArea){
            this.area = initArea;
            position  = new PVector(random(0, initArea.width),random(0, initArea.height));
            velocity  = new PVector(random(-1, 1),random(-1, 1));
            radius    = random(MIN_RADIUS, MAX_RADIUS);
            hue       = 0;
        }

        public void checkBounds(){
            // simple bounce when beyond bounds.  only accomplishes 90 or 180
            // degree turns.
            if(position.x > area.width) {
                position.x = area.width;
                velocity.x = -velocity.x;
            }
            if(position.x < 0) {
                position.x = 0;
                velocity.x = -velocity.x;
            }
            if(position.y > area.height) {
                position.y = area.height;
                velocity.y = -velocity.y;
            }
            if(position.y < 0) {
                position.y = 0;
                velocity.y = -velocity.y;
            }
        }
    }
    
    class MetaballGroup {
        
    }
}