package net.electroland.fish.boids;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.vecmath.Vector3f;

import com.sun.opengl.util.GLUT;

import net.electroland.broadcast.fish.Fish;
import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Maps;
import net.electroland.fish.core.Pond;
import net.electroland.fish.core.ContentLayout.Slot;
import net.electroland.fish.forces.ForceMap;
import net.electroland.fish.forces.SinkOnWave;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.Content;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;

public class VideoFish extends Boid {

	//ds on site change
	//public static final long SEND_CONTENT_DELAY = 200;
	public static final long SEND_CONTENT_DELAY = 300;
	public long killTime;
	public long contentTime;
	public Content content;
	boolean sent;

	float goalDisplayLocationX;
	float goalDisplayLocationY;
	Bounds goalBounds;

	public Slot slot;


	public VideoFish(Pond pond, float scale, int flockId, Content c) {
		super(pond, scale, 14, false);
		this.offScreen = false;
		this.species = flockId;
		this.content = c;
		setSize(70);
		goalBounds = new Bounds();

		MaxSpeed maxSpeed;
		add("maxSpeed", maxSpeed = new MaxSpeed(1,20));
		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("VideoFishMaxSpeed", 5), 1));

		add("rotate", new ForceMap(5, this, Maps.FORCES,.05f));
//		add("wave", new Wave(1000000,this, Util.plusOrMinus(2000, 500)));
		add("sinkOnWave", new SinkOnWave(1.0f, this, 3000));


	}

	public void show(Vector3f pos, float headX, float headY) {
		position = pos;
		this.heading.x = headX;
		this.heading.y = headY;


		killTime = pond.CUR_TIME + content.duration + 100 + SEND_CONTENT_DELAY;
		contentTime = pond.CUR_TIME + SEND_CONTENT_DELAY;

		broadcastFish = new net.electroland.broadcast.fish.Fish(Integer.toString(id), 14,
				heading.x,heading.y, scale,
				vectorToAngle(heading.x, heading.y), 
				0, 0, 
				0,0,0,null);

		//	broadcastFish.movieFileName = content.name; wait 100 ms to do this for flash to catch up

		this.broadcastFish.orientation = vectorToAngle(headX, headY);
		broadcastFishList.add(broadcastFish);


		this.setVelocity(new Vector3f(5.0f, 5.0f, 0.0f));



		pond.add(this);		
		System.out.println("showing " + content.name);
	}

	public void setAngle(float a) {
		broadcastFish.orientation = a;
	}

	public void setGoalDisplayLocation(float x, float y) {

		goalDisplayLocationX = x;
		goalDisplayLocationY = y;
		goalBounds.setLeft(goalDisplayLocationX - content.halfWidth);
		goalBounds.setRight(goalDisplayLocationX + content.halfWidth);
		goalBounds.setTop(goalDisplayLocationY - content.halfHeight);
		goalBounds.setBottom(goalDisplayLocationY + content.halfHeight);

	}
	
    int font = GLUT.BITMAP_HELVETICA_10;
    public final GLUT glut = new GLUT();
	public void draw(GLAutoDrawable drawable, GL gl) {
		super.draw(drawable, gl);
		gl.glColor3f(1,0,0);

		gl.glRasterPos3f(position.x, position.y, 0);
		glut.glutBitmapString(font, "VID");
        
	}

	public boolean sendContent = true;

	public void move() {
		super.move();
		if(contentTime <= pond.CUR_TIME) {
			if(sendContent) {
				broadcastFish.movieFileName = content.name;
				sendContent = false;
			}
		} if(killTime <= pond.CUR_TIME) {
			if(this.broadcastFish.state == Fish.NOSTATE) { // if kill already broadcasted
				pond.remove(this);
				broadcastFishList.remove(broadcastFish);
			} else if(this.broadcastFish.state != -1) {
				//		System.out.println("killing video fish");
				this.broadcastFish.state = -1;
				this.broadcastFish.accent = 14; // state = die();
				slot.vid = null;
			} 
		}

	}
}
