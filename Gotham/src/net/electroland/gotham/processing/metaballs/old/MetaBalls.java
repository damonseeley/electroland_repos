package net.electroland.gotham.processing.metaballs.old;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import net.electroland.gotham.processing.GothamPApplet;

import processing.core.PVector;

/**
 * Adapted from http://www.openprocessing.org/sketch/13250
 */
public class MetaBalls extends GothamPApplet {

    private static final long serialVersionUID = 7554567212859904536L;

    // we want the lava to stay on screen
    final Rectangle BOUNDARY    = new Rectangle(80, 70, 621 - 80, 435 - 70);
    final float BORDER_CLOSENESS = 1.5f; // higher = more of the ball can go beyond the border

    final int   COARSENESS      = 10;
    final int   NUM_BALLS       = 3;
    final int   MIN_RADIUS      = 100;    // initial ball radius min
    final int   MAX_RADIUS      = 200;    // initial ball radius max

    final float FRICTION        =  0.9f; // higher = less
    final float P_FRICTION      =  1.1f;
    final float MAX_VEL         = 10;     // higher = faster
    final float MIN_VEL         = .5f;    // higher = faster
    final float COHESION        = .003f;  // higher = more
    final float P_COHESION      = -COHESION;

    private List<Metaball> balls;
    private Point mainPresence;

    @Override
    public void setup(){

        colorMode(HSB, 360, 100, 100);
        background(0);

        balls = new ArrayList<Metaball>();
        balls.add(new Metaball(0,  new Dimension(200,200), this.getSyncArea()));
        balls.add(new Metaball(40, new Dimension(200,200), this.getSyncArea()));
        balls.add(new Metaball(80, new Dimension(300,300), this.getSyncArea()));
    }

    float max = 0;
    @Override
    public void drawELUContent() {

        PVector center = new PVector(0, 0, 0);

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
                    ball.distance = sqrt(sq(i - ball.position.x) + sq(j - ball.position.y));
                    sum += ball.radius / ball.distance;
                }

                float brightness = (sum * sum * sum) / 5;
                fill(0, 100, brightness);
                this.rect(i, j, COARSENESS, COARSENESS);
            }
        }
        // show where the mouse was clicked
        if (mainPresence != null){
            stroke(255, 255, 255);
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
        Dimension range;
        Dimension size;
        float distance;
        int hue;

        public Metaball(int hue, Dimension size, Dimension range){
            this.hue = hue;
            this.range = range;
            this.size = size;
            position  = new PVector(random(0, range.width),random(0, range.height));
            velocity  = new PVector(random(-1, 1),random(-1, 1));
            radius    = size.width;
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