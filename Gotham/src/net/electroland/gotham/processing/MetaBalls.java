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
    final float BORDER_CLOSENESS = 1.5f; // higher = more of the ball can go beyond the border

    final int   COARSENESS      = 5;
    final int   NUM_BALLS       = 3;
    final int   MIN_RADIUS      = 100;    // initial ball radius min
    final int   MAX_RADIUS      = 200;   // initial ball radius max

    final float FRICTION        =  0.99f;  // higher = less
    final float P_FRICTION      =  1.25f;
    final float MAX_VEL         = 10;    // higher = faster
    final float MIN_VEL         = 1;    // higher = faster
    final float COHESION        =   .003f;  // higher = more
    final float P_COHESION      = -COHESION;

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

        PVector center = mainPresence == null ? new PVector(0, 0, 0) : 
                                                new PVector(mainPresence.x, mainPresence.y, 0);

        //update meta ball positions
        for (Metaball ball : balls){
            center.add(ball.position);
            ball.velocity.mult(mainPresence != null ? P_FRICTION : FRICTION);
            ball.velocity.limit(MAX_VEL);
            if (ball.velocity.mag() < MIN_VEL)
                ball.velocity.setMag(MIN_VEL);
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
        noStroke();
        for(int i = 0; i < this.getSyncArea().width; i+= COARSENESS) {
            for(int j = 0; j < this.getSyncArea().height; j+= COARSENESS) {
                float sum = 0;
                for (Metaball ball : balls){
                    // radius divided by distance from this pixel to the center of the ball.  meaning max = ball.radius 
                    ball.pixelImpact = ball.radius / sqrt(sq(i - ball.position.x) + sq(j - ball.position.y));
                    sum += ball.pixelImpact;
                }
                float color = (sum * sum * sum) / 3;
                fill(0, 100, color);
//                fill(color, color, color);
                this.rect(i, j, COARSENESS, COARSENESS);
            }
        }

        // show where the mouse was clicked
        if (mainPresence != null){
            stroke(0, 0, 100);
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
            if(position.x + (radius / BORDER_CLOSENESS) > BOUNDARY.x + BOUNDARY.width) {
                position.x = BOUNDARY.x + BOUNDARY.width - (radius / BORDER_CLOSENESS);
                velocity.x = -velocity.x;
            }
            if(position.x -(radius / BORDER_CLOSENESS) < BOUNDARY.x) {
                position.x = BOUNDARY.x + (radius / BORDER_CLOSENESS);
                velocity.x = -velocity.x;
            }
            if(position.y + (radius / BORDER_CLOSENESS)> BOUNDARY.y + BOUNDARY.height) {
                position.y = BOUNDARY.y + BOUNDARY.height - (radius / BORDER_CLOSENESS);
                velocity.y = -velocity.y;
            }
            if(position.y - (radius / BORDER_CLOSENESS) < BOUNDARY.y) {
                position.y = BOUNDARY.y + (radius / BORDER_CLOSENESS);
                velocity.y = -velocity.y;
            }
        }
    }
}