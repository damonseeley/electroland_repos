package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs2 extends GothamPApplet {

    public static final float FRICTION        = .999f; // higher = less
    public static final float MAX_VEL         = 1.0f;     // higher = faster
    public static final float MAX_RUN_VEL     = 2.0f;     // higher = faster
    public static final float MIN_VEL         = .75f;     // higher = faster
    public static final float COHESION        = .05f;  // higher = more
    public static final float SQUISHINESS     = 50;    // higher = more
    public static final float ELLIPSE_SCALE   = 2.0f;  // percent
    public static final Rectangle RANGE = new Rectangle(80, 70, 621 - 80, 435 - 70);

    private List <MetaballGroup>groups;
    private Point mainPresence;

    @Override
    public void setup(){

    	// set background color
    	background(0);

    	// groups of balls
        groups = new ArrayList<MetaballGroup>();

        MetaballGroup red = new MetaballGroup(new Rectangle(80 - 50, 70 - 50, 580 + 75, 384 + 75), new Color(255, 0, 0), SQUISHINESS);
        groups.add(red);
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup orange = new MetaballGroup(new Rectangle(80 - 50, 70 - 50, 580 + 75, 384 + 75), new Color(255, 127, 0), SQUISHINESS);
        groups.add(orange);
        orange.add(new Metaball(75 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup purple = new MetaballGroup(new Rectangle(80 - 50, 70 - 50, 580 + 75, 384 + 75), new Color(128, 0, 128), SQUISHINESS);
        groups.add(purple);
        purple.add(new Metaball(75 * ELLIPSE_SCALE));
        purple.add(new Metaball(100 * ELLIPSE_SCALE));
        purple.add(new Metaball(100 * ELLIPSE_SCALE));

        // probably should be in ball constructors
        for (MetaballGroup group : groups){
        	for (Metaball ball : group.balls){
                ball.position  = new PVector(random(RANGE.x + SQUISHINESS, RANGE.width + RANGE.x - (2 * SQUISHINESS)),
                        random(RANGE.y + SQUISHINESS, RANGE.height + RANGE.y - (2 * SQUISHINESS)));
                ball.velocity = new PVector(random(-1,1), random(-1,1));
        	}
        }
    }

    @Override
    public void drawELUContent() {

        // move balls
    	for (MetaballGroup group : groups){

    		// cohesion, etc. are per group
    		PVector center = new PVector(0, 0, 0);

    		for (Metaball ball : group.balls){

    			// mousePressed = proxy for presence grid repelling
    		    if(mousePressed) {
    		    	ball.repell(new PVector(mouseX, mouseY, 0));
    		    	ball.spaceCheck(groups);
    		    }

    		    center.add(ball.position);
                ball.velocity.mult(FRICTION);
                ball.velocity.limit(mainPresence != null ? MAX_RUN_VEL : MAX_VEL);
                if (ball.velocity.mag() < MIN_VEL)
                    ball.velocity.setMag(MIN_VEL);
            }

    		center.div(group.balls.size());

            for (Metaball ball : group.balls){
                PVector c = PVector.sub(center, ball.position);
                c.normalize();
                c.mult(COHESION);
                ball.velocity.add(c);
                ball.position.add(ball.velocity);
            }
            
        	group.checkBounds();
    	}

        // erase screen
        this.fill(0);
        this.rect(0, 0, width, height);

        
        // render each group's bounding box
        for (MetaballGroup group : groups){
        	this.stroke(255,255,255);
        	this.noFill();
        	this.rect(group.range.x, group.range.y, group.range.width, group.range.height);
        }
        
        // render groups of balls
    	for (MetaballGroup group : groups){
	        for (Metaball ball : group.balls){
	            this.noStroke();
	            this.fill(group.color.r, group.color.g, group.color.b, 255/2);
	            this.ellipse(ball.position.x, ball.position.y, ball.width(), ball.height());
	        }
    	}
    	// blur the whole thing
        //filter(BLUR, 3);

        // mouse cursor
        if (mainPresence != null){
            fill(255, 255, 255);
            ellipse(mainPresence.x, mainPresence.y, 60, 40);
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

class MetaballGroup {
	
	Rectangle range;
	float squishiness;
    PVector position;
    PVector velocity;
    Color color;
    List <Metaball>balls;

    public MetaballGroup(Rectangle range, Color color, float squishiness){
    	this.range = range;
    	this.color = color;
    	this.squishiness = squishiness;
    	balls = new ArrayList<Metaball>();
    }
    
    public void add(Metaball ball){
    	ball.group = this;
    	balls.add(ball);
    }
    
    public void checkBounds(){
    	for (Metaball ball : balls){
    		ball.checkBounds(range,  squishiness);
    	}
    }
}

class Metaball {

    float radius;

    PVector position;
    PVector velocity;
    float xsquish = 0;
    float ysquish = 0;
    MetaballGroup group;

    public Metaball(float radius){
        this.radius = radius;
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
        // squish (needs work)
        /*
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
        */
    }
    
    public void repell(PVector point){
    	
    	float mDistance = PVector.dist(position, point);
    	PVector repel = PVector.sub(position, point);
        repel.normalize();
        repel.mult(60 / (mDistance * mDistance));
        velocity.add(repel);
    	
    }

    public void spaceCheck(List<MetaballGroup> groups){
    	for (MetaballGroup group : groups) {
    		if (group != this.group) {
    	    	for (Metaball ball : group.balls){
    	    		if (ball.overlaps(this) || ball.group != this.group){
    	    			this.repell(ball.position);
    	    		}
    	    	}
    		}
    	}
    }
    
    public boolean overlaps(Metaball ball){
    	float distance = ball.position.dist(this.position);
    	return distance < ball.radius || distance < this.radius;
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