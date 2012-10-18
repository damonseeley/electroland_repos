package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.assets.FastBlur;

import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs2 extends GothamPApplet {

	public static final int NONE 				= 0;
	public static final int MOUSE 				= 1;
	public static final int TRACK	 			= 2;

    private int mode = MOUSE; // set the reaction mode
	
	
	public static final float FRICTION        	= .999f;  // higher = less .999
    public static final float MAX_VEL         	= 0.75f;  // base velocity with no interaction or tracks higher = faster
    // push vars
    public static final float MAX_RUN_VEL     	= 1000.0f;  // max velocity when mouse is down or presence is felt.  //30
    public static final float REPELL_FORCE   	= 1000; // repell force of mouse or track (higher = more)
    // ball group props
    public static final float BALL_REPELL_FORCE	= 20;	  // group to group repell force (higher = more)
    public static final float COHESION        	= .005f;   // higher = more
    // ball scale
    public static final float ELLIPSE_SCALE   	= 2.0f;   // percent

    // don't touch:
    public static final float SQUISHINESS     	= 50;     // higher = more
    public static final float MIN_VEL         	= .75f;   // higher = faster

    
    private List <MetaballGroup>groups;
    
    private Point mainPresence;
    
	private GridData gd;

    @Override
    public void setup(){

    	// set background color
    	background(0);

    	// groups of balls
        groups = new ArrayList<MetaballGroup>();

        MetaballGroup red = new MetaballGroup(new Rectangle(-100, -100, this.getSyncArea().width + 100, this.getSyncArea().height +100), new Color(255, 0, 0), SQUISHINESS);
        groups.add(red);
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup orange = new MetaballGroup(new Rectangle(-100, -100, this.getSyncArea().width + 100, this.getSyncArea().height +100), new Color(255, 127, 0), SQUISHINESS);
        groups.add(orange);
        orange.add(new Metaball(75 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));
        orange.add(new Metaball(75 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));
        orange.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup purple = new MetaballGroup(new Rectangle(0, 0, this.getSyncArea().width, this.getSyncArea().height), new Color(128, 0, 128), SQUISHINESS);
        groups.add(purple);
        purple.add(new Metaball(25 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(25 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));

        // probably should be in ball constructors
        for (MetaballGroup group : groups){
        	for (Metaball ball : group.balls){
                ball.position  = new PVector(random(0, this.getSyncArea().width), random (0, this.getSyncArea().height));
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
    		    if(mode == MOUSE && mousePressed) {
    		    	ball.repell(new PVector(mouseX, mouseY, 0), REPELL_FORCE);
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

        // presence
        if (mode == MOUSE && mousePressed){
            fill(255, 255, 255);
            //ellipse(mouseX, mouseY, 60, 40);
            ellipse(mouseX, mouseY, 140, 100);
        } else if (mode == TRACK) {
        	
        	
        	
        	
        	// RENDER THE GRID CELLS AS VISUAL WHITE THINGS
        	
        	int insetLeft = 80;
    		int insetTop = 70;
    		double dilate = 6.0;
    		
    		fill(color(0, 0, 50),8); //fill with a light alpha white
    		rect(0,0,this.getSyncArea().width,this.getSyncArea().height); //fill the whole area

    		if (gd != null){
    			int gridXStart = 2;
    			int gridXMax = 70;
    			int gridYStart = 2; // test start inset on top
    			int gridYMax = 25; //all of the height
    			
    			int hShift = -0;

    			int cellWidth = (this.getSyncArea().width-(insetLeft*2))/(gridXMax-gridXStart);
    			int cellHeight = (this.getSyncArea().height-(insetTop*2))/(gridYMax-gridYStart);

    			for(int y = gridYStart; y < gridYMax; y++) {
    				for(int x = gridXStart; x < gridXMax; x++) {
    					if (gd.getValue(x, y) > 0){
    						fill(color(255, 255, 255));
    						ellipse(this.getSyncArea().width-(insetLeft*2)+hShift-y*cellHeight+insetLeft, this.getSyncArea().height-(insetTop*2)-x*cellWidth+insetTop, (int)(cellHeight*dilate), (int)(cellWidth*dilate)); //rotated

    					}
    				}
    			}
    		}
        	
        }
        
        
        
        
        
        
        
        
        loadPixels();
		FastBlur.performBlur(pixels, width, height, floor(20));
		updatePixels();
		
    }
    
	@Override
	public void handle(GridData t) {
		gd = t;
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
    
    public void repell(PVector point, float force){
    	
    	float mDistance = PVector.dist(position, point);
    	PVector repel = PVector.sub(position, point);
        repel.normalize();
        repel.mult(force / (mDistance * mDistance));
        velocity.add(repel);
    	
    }

    public void spaceCheck(List<MetaballGroup> groups){
    	for (MetaballGroup group : groups) {
    		if (group != this.group) {
    	    	for (Metaball ball : group.balls){
    	    		if (ball.overlaps(this) || ball.group != this.group){
    	    			this.repell(ball.position, Metaballs2.BALL_REPELL_FORCE);
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