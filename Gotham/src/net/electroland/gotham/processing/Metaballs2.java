package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs2 extends GothamPApplet {

    public static final float FRICTION        =  0.9f; // higher = less
    public static final float MAX_VEL         = 10;    // higher = faster
    public static final float MIN_VEL         = .5f;   // higher = faster
    public static final float COHESION        = .003f; // higher = more
    public static final float P_FRICTION      =  1.1f;
    public static final float P_COHESION      = -COHESION;
    public static final float SQUISHINESS     = 50;    // higher = more
    public static final Rectangle RANGE = new Rectangle(80, 70, 621 - 80, 435 - 70);

    private List <Metaball>balls;
    private Point mainPresence;

    @Override
    public void setup(){
        background(0);
        balls = new ArrayList<Metaball>();
        balls.add(new Metaball(300, new Color(255,0,0)));
        balls.add(new Metaball(200, new Color(255, 97, 3)));
        balls.add(new Metaball(200, new Color(255,0,0)));
 
        for (Metaball ball : balls){
            ball.position  = new PVector(random(RANGE.x + SQUISHINESS, RANGE.width + RANGE.x - (2 * SQUISHINESS)),
                                         random(RANGE.y + SQUISHINESS, RANGE.height + RANGE.y - (2 * SQUISHINESS)));
            ball.velocity = new PVector(random(-1,1), random(-1,1));
        }
    }

    @Override
    public void drawELUContent() {
        // move balls

        PVector center = new PVector(0, 0, 0);
        for (Metaball ball : balls){
            center.add(ball.position);
            ball.velocity.mult(mainPresence != null ? P_FRICTION : FRICTION);
            ball.velocity.limit(MAX_VEL);
            if (ball.velocity.mag() < MIN_VEL)
                ball.velocity.setMag(MIN_VEL);
        }
        center.div(balls.size());

        for (Metaball ball : balls){
            PVector c = PVector.sub(center, ball.position);
            c.normalize();
            c.mult(mainPresence != null ? P_COHESION : COHESION);
            ball.velocity.add(c);
            ball.position.add(ball.velocity);

            ball.checkBounds(RANGE, SQUISHINESS);
        }

        // erase screen
        this.fill(0);
        this.rect(0, 0, width, height);

        // render balls
        for (Metaball ball : balls){
            this.noStroke();
            this.fill(ball.color.r, ball.color.g, ball.color.b, 255/2);
            this.ellipse(ball.position.x, ball.position.y, ball.width(), ball.height());
        }
        filter(BLUR, 3);

        if (mainPresence != null){
            stroke(255, 255, 255);
            line(mainPresence.x - 10, mainPresence.y, mainPresence.x + 10, mainPresence.y);
            line(mainPresence.x, mainPresence.y - 10, mainPresence.x, mainPresence.y + 10);
        }
    }

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
}

class Metaball {

    float radius;
    Color color;

    PVector position;
    PVector velocity;
    float xsquish = 0;
    float ysquish = 0;

    public Metaball(float radius, Color color){
        this.radius = radius;
        this.color = color;
    }

    public void checkBounds(Rectangle range, float squishiness){
        // bounce
        if (left() < range.x){
            setLeft(range.x);
            velocity.x = -velocity.x;
        } else if (right() > range.width + range.x){
            setRight(range.width + range.x);
            velocity.x = -velocity.x;
        }
        if (top() < range.y){
            setTop(range.y);
            velocity.y = -velocity.y;
        } else if (bottom() > range.height + range.y){
            setBottom(range.height + range.y);
            velocity.y = -velocity.y;
        }
        // squish
        if (left() < range.x + squishiness){
            xsquish -= velocity.x;
        } else if (right() > range.width + range.x - squishiness){
            xsquish += velocity.x;
        }
        if (top() < range.y + squishiness){
            ysquish -= velocity.y;
        } else if (bottom() > range.height + range.y - squishiness){
            ysquish += velocity.y;
        }

    }

    public float width(){
        return radius - xsquish;
    }
    public float height(){
        return radius - ysquish;
    }

    public float left(){
        return position.x - width() / 2;
    }
    public float right(){
        return position.x + width() / 2;
    }
    public float top(){
        return position.y - height() / 2;
    }
    public float bottom(){
        return position.y + height() / 2;
    }

    public void setLeft(float left){
        position.x = width() / 2 + left; 
    }
    public void setRight(float right){
        position.x = right - width() / 2;
    }
    public void setTop(float top){
        position.y = top + height() / 2;
    }
    public void setBottom(float bottom){
        position.y = bottom - height() / 2;
    }
}

class Color {
    float r, g, b;
    public Color(float r, float g, float b){
        this.r = r;
        this.g = g;
        this.b = b;
    }
}