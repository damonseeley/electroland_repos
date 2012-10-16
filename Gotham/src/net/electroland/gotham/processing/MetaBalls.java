package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import processing.core.PVector;

/**
 * Adapted from http://www.openprocessing.org/sketch/13250
 */
public class MetaBalls extends GothamPApplet {

    private static final long serialVersionUID = 7554567212859904536L;

    // we want the lava to stay on screen
    final Rectangle BOUNDARY    = new Rectangle(60,60,580,390);

    final int   COARSNESS       = 5;
    final int   NUM_BALLS       = 3;
    final int   MIN_RADIUS      = 200;    // initial ball radius min
    final int   MAX_RADIUS      = 400;   // initial ball radius max

    final float FRICTION        =   .75f;  // higher = less
    final float P_FRICTION      =  1.0f;
    final float MAX_VEL         = 15;    // higher = faster
    final float COHESION =   .05f;  // higher = more
    final float P_COHESION      = -COHESION;


    // TODO: should cycle hue and saturation from an image file
    private int hue             = 0;
    private int saturation      = 100;


    // might be nice to clean these up by putting them in a Collection of Metaballs.
    private List<Metaball> balls;
    private Point mainPresence;


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
            ball.velocity.mult(mainPresence != null ? P_FRICTION : FRICTION);
            ball.velocity.limit(MAX_VEL);
        }

        center.div(NUM_BALLS);

        for (Metaball ball : balls){

            PVector c = PVector.sub(center, ball.position);
            c.normalize();
            c.mult(mainPresence != null ? P_COHESION : COHESION);
            ball.velocity.add(c);
            ball.position.add(ball.velocity);

            ball.checkBounds();
        }

        // render
        //hue++; // cheaply cycle hue

        noStroke();
        for(int i = 0; i < this.getSyncArea().width; i+= COARSNESS) {
            for(int j = 0; j < this.getSyncArea().height; j+= COARSNESS) {
                float sum = 0;
                for (Metaball ball : balls){
                    // radius divided by distance from this pixel to the center of the ball.  meaning max = ball.radius 
                    ball.pixelImpact = ball.radius / sqrt(sq(i - ball.position.x) + sq(j - ball.position.y));
                    sum += ball.pixelImpact;
                }
                fill(hue % 360, saturation, (sum * sum * sum) / 3);
                this.rect(i, j, COARSNESS, COARSNESS);
            }
        }

        // show where the mouse was clicked
        if (mainPresence != null){
            stroke(0, 0, 255);
            line(mainPresence.x - 10, mainPresence.y, mainPresence.x + 10, mainPresence.y);
            line(mainPresence.x, mainPresence.y - 10, mainPresence.x, mainPresence.y + 10);
        }
    }


    Point down;
    boolean on;
    public void mousePressed() {
        mainPresence = new Point(mouseX, mouseY);
    }

    public void mouseDragged(){
        mainPresence = new Point(mouseX, mouseY);
    }

    public void mouseReleased(){
        mainPresence = new Point(mouseX, mouseY);
        mainPresence = null;
    }

    class Metaball {

        float radius;
        PVector position;
        PVector velocity;
        Dimension area;
        float pixelImpact;

        public Metaball(Dimension initArea){
            this.area = initArea;
            position  = new PVector(random(0, initArea.width),random(0, initArea.height));
            velocity  = new PVector(random(-1, 1),random(-1, 1));
            radius    = random(MIN_RADIUS, MAX_RADIUS);
        }

        public void checkBounds(){
            // simple bounce when beyond bounds.  only accomplishes 90 or 180
            // degree turns.
            if(position.x > BOUNDARY.x + BOUNDARY.width) {
                position.x = BOUNDARY.x + BOUNDARY.width;
                velocity.x = -velocity.x;
            }
            if(position.x < BOUNDARY.x) {
                position.x = BOUNDARY.x;
                velocity.x = -velocity.x;
            }
            if(position.y > BOUNDARY.y + BOUNDARY.height) {
                position.y = BOUNDARY.y + BOUNDARY.height;
                velocity.y = -velocity.y;
            }
            if(position.y < BOUNDARY.y) {
                position.y = BOUNDARY.y;
                velocity.y = -velocity.y;
            }
        }
    }
}