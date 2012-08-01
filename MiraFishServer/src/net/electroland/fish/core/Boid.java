package net.electroland.fish.core;

import java.awt.Point;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.vecmath.Vector3f;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.regions.TriggerListener;
import net.electroland.fish.ui.Drawable;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.SortedVector;

public  class  Boid implements Drawable, TriggerListener {

	public boolean offScreen = true;

	public final static float RADS_TO_DEGREES = 360.0f / (2.0f * 3.14159265358979f);

	public static final int FLASH_DEPTH_LIMIT = FishProps.THE_FISH_PROPS.getProperty("flashDepthLimit", 50);

	public static int curSubFlockId = 0;
	public static int FLOCKSIZE = 2;
	public static int curSubFlockCnt=0;

	public float scaleCompression = -1;

	public static void setSubFlock(Boid b) {
		if(curSubFlockCnt >= FLOCKSIZE) {
			curSubFlockCnt = 0;
			curSubFlockId++;
		}
		b.subFlockId = curSubFlockId;
		curSubFlockCnt++;
	}

	public static void resetSubFlockCount(int max) {
		FLOCKSIZE = max;
		curSubFlockCnt = 0;
	}


	// ID is just for debugging;
	public static int CURID = 0;
	public int id = CURID++;

	public Pond pond;

	public net.electroland.broadcast.fish.Fish broadcastFish;

	public static final List<net.electroland.broadcast.fish.Fish> broadcastFishList = new LinkedList<net.electroland.broadcast.fish.Fish>();

	protected int species = 0;
	public int subFlockId = 0;

	protected boolean isTouchable = true;

	protected boolean isVisibleToBoids = true;

	public boolean isTouched = false;

	protected float visionRadius = 300.0f;	
	public  float size = 1; // size of fish
	protected float scale = 1;

	protected Point scpacialGridLoc = new Point();

	protected Vector3f proposedPosition = new Vector3f();
	protected Vector3f position = new Vector3f();
	protected Vector3f heading = new Vector3f(1f,0f,0f);
	protected Vector3f left = new Vector3f(0f,1f,0f);
	protected Vector3f velocity = new Vector3f();
	protected Vector3f scaledVelocityPerFrame = new Vector3f();

	protected Vector3f deltaVelocity = new Vector3f();
	protected Vector3f oldPosition = new Vector3f(); // used to see if crossed barrier

	protected long nextTouchTime = 0;
	protected final long touchTimeOut = FishProps.THE_FISH_PROPS.getProperty("touchTimeOut", 500);



	protected ConcurrentHashMap<String, Behavior> behaviors = new ConcurrentHashMap<String, Behavior>();
	protected ConcurrentHashMap<String, Force> forces = new ConcurrentHashMap<String, Force>();
	protected ConcurrentHashMap<String, Constraint> constraintMap =  new ConcurrentHashMap<String, Constraint>();
	protected SortedVector<Constraint> constraintVec = new SortedVector<Constraint>();

	private Vector3f drawHeading = new Vector3f(); // just a cache to prevent recreate object each frame

	private float depthColorScale = .5f;


	public Boid(Pond pond, float scale, int flockId) {
		this(pond, scale, flockId, new Vector3f(), 1, 0, true);
	}
	public Boid(Pond pond, float scale, int flockId, boolean createBroadcastFish) {
		this(pond, scale, flockId, new Vector3f(), 1, 0, createBroadcastFish);
	}


	public Boid(Pond pond, float scale, int flockId,  Vector3f vec, float headingx, float headingy, boolean createBroadcastFish) {
		this.pond = pond;
		this.scale = scale;
		this.size = 5;
		this.species = flockId;
		this.position = vec;

		depthColorScale = 1.0f/ pond.bounds.getDepth();
		this.heading.x = headingx;
		this.heading.y = headingy;

		if(createBroadcastFish) {
			broadcastFish = new net.electroland.broadcast.fish.Fish(Integer.toString(id), flockId,
					vec.x,vec.y, scale,
					vectorToAngle(headingx, headingy), 
					0, 0, 
					0, //state
					net.electroland.broadcast.fish.Fish.NOSTATE, //accent
					0, // frame
					null);
			broadcastFishList.add(broadcastFish);
		}

		this.setSize(3);


	}

	public Vector3f getLeft() {
		return left;
	}

	public void add(String s, Behavior b) {
		behaviors.put(s, b);
	}

	public void removeBehavior(String s) {
		behaviors.remove(s);
	}

	public Behavior getBehavior(String s) {
		return behaviors.get(s);

	}

	public void add(String s, Force f) {
		forces.put(s, f);
	}


	public Force getForce(String s) {
		return forces.get(s);
	}

	public void removeForce(String s) {
		forces.remove(s);
	}

	public void add(String s, Constraint c) {
		constraintMap.put(s, c);
		constraintVec.add(c);
	}

	public void removeConstraint(String s) {
		Constraint c = constraintMap.get(s);
		if(c != null)
			constraintVec.remove(c);
	}

	public Constraint getConstraint(String s) {
		return  constraintMap.get(s);
	}

	public int getFlockId() {
		return species;
	}
	public void setFlockId(int flockId) {
		this.species = flockId;
	}
	public float getVisionRadius() {
		return visionRadius;
	}

	float halfVisionRadius = visionRadius *.5f;
	public void setVisionRadius(float visionRadius) {
		this.visionRadius = visionRadius;
		halfVisionRadius = visionRadius *.5f;
	}
	public Vector3f getPosition() {
		return position;
	}
	public void setPosition(Vector3f position) {
		this.position = position;
	}
	public Vector3f getHeading() {
		return heading;
	}
	public void setHeading(Vector3f heading) {
		this.heading = heading;
	}

	public void setVelocity(Vector3f velocity) {
		this.velocity = velocity;
	}

	public Vector3f getVelocity() {
		return velocity;
	}
	public Point getScpacialGridLoc() {
		return scpacialGridLoc;
	}
	public Vector3f getOldPosition() {
		return oldPosition;
	}

	public void teleport(Vector3f newPosition) {
		oldPosition.set(newPosition);
		position.set(newPosition);	
	}

	public void addVelocity(float x, float y, float z) {
		velocity.x +=x;
		velocity.y +=y;
	}
	public void addVelocity(Vector3f vel) {
		velocity.add(vel);
	}


	public void reactToBoids(Vector<Boid> boids) {

		//for(VisionTransition vt : visionTransitions ) {
		//vt.see(boids);
		//}

		for(Behavior behavior :  behaviors.values()) {
			for(Boid boid : boids) {
				behavior.see(boid);				
			}
		}
	}


	public void applyForces() {
		if(offScreen) { // no forces or behavoirs while off screen, just drift onto screen
			return;
		}
		float totalWeight = 0;
		Vector3f tmp = new Vector3f();
		for(Force f : forces.values()) {
			if(f.isEnabled()) {
				ForceWeightPair fwp = f.getForce();
				if(f.weight != 0) {
					tmp.set(fwp.force);
					tmp.scale(fwp.weight);					
					deltaVelocity.add(tmp);		
					totalWeight += fwp.weight;				

				}
			}
		}
		for(Behavior behavior : behaviors.values()) {
			if(behavior.isEnabled()) {
				ForceWeightPair fwp = behavior.getForce();
				if(fwp.weight != 0) {
					tmp.set(fwp.force);
					tmp.scale(fwp.weight);
					deltaVelocity.add(tmp);		
					totalWeight += behavior.weight;				
				}
			}
		}
		if(totalWeight != 0) {
			deltaVelocity.scale(1.0f/totalWeight);

		}

	}

	public boolean isBigFish = false;
	public Vector3f oneCellBack = new Vector3f();
	public int cellLookBackCount = FishProps.THE_FISH_PROPS.getProperty("bigFishLookBack", 3);
	public void senceTouch() {
		isTouched = false;
		if(nextTouchTime < pond.CUR_TIME) {
			if(pond.grid.cells[scpacialGridLoc.x][scpacialGridLoc.y].isTouched) {
				nextTouchTime = pond.CUR_TIME + touchTimeOut;
				isTouched = true;
			} else if(isBigFish) {
				int i = 0;
				while((i++ < cellLookBackCount) && ! isTouched) {
					oneCellBack.set(heading);
					oneCellBack.scale(-pond.grid.cellWidth);
					oneCellBack.add(position);				
					Point ptBack = pond.grid.getGridLocation(oneCellBack);
					if(pond.grid.cells[ptBack.x][ptBack.y].isTouched) {
						isTouched = true;
					} else {
						isTouched = false;
					}
				}
			} else {
				isTouched = false;
			}
		} else {
			isTouched = false;
		}
		if(isTouched) {
			touched();
		}
	}

	public void touched() {
		broadcastFish.accent = 0; // touch accent
	}

	public void move() {
		move(true);
	}

	public void move( boolean setOrientation) {

		left.set(-heading.y, heading.x, 0);
		left.normalize();

		velocity.add(deltaVelocity);
		deltaVelocity.set(0f,0f,0f);

		scaledVelocityPerFrame.set(velocity);
		scaledVelocityPerFrame.scale(pond.CURFRAME_TIME_SCALE);

		oldPosition.set(position);		

		proposedPosition = new Vector3f(position);

		proposedPosition.add(scaledVelocityPerFrame);

		scpacialGridLoc = pond.grid.move(scpacialGridLoc, proposedPosition, this);

		for(Constraint c : constraintVec) {
			if(c.isEnabled) {
				if(c.modify(proposedPosition, velocity, this))
					scpacialGridLoc = pond.grid.move(scpacialGridLoc, proposedPosition, this);
			}
		}

		position.set(proposedPosition);


		broadcastFish.x = position.x;
		broadcastFish.y =  position.y;
		broadcastFish.depth = (int) (position.z * 200);
		broadcastFish.depth = (broadcastFish.depth < FLASH_DEPTH_LIMIT) ? FLASH_DEPTH_LIMIT : broadcastFish.depth;
		broadcastFish.speed =  scaledVelocityPerFrame.length();


		if(scaleCompression > 0) {
			float diff = position.z - 1.0f;
			diff *= scaleCompression;
			diff += 1;

			diff = (diff >= 2) ? 2 : diff;
			diff = (diff <= 0) ? 0 : diff;

			broadcastFish.scale =  diff;
		}

		if(setOrientation) {
			broadcastFish.orientation = vectorToAngle(heading.x, heading.y);			
		}

	}

	public static float vectorToAngle(float x, float y) {
		if(x == 0) {
			if(y > 0) {
				return 90.0f;
			} else {
				return -90.0f;
			}
		} if(x > 0) {
			return (float) (RADS_TO_DEGREES * Math.atan(y / x));											
		} else {
			if(y >= 0) {
				return (float) (RADS_TO_DEGREES * Math.atan(y / x)) + 180;															
			} else {
				return (float) (RADS_TO_DEGREES * Math.atan(y / x)) - 180;															
			}
		}

	}


	public void draw(GLAutoDrawable drawable, GL gl) {
		drawHeading.set(heading);
		drawHeading.scale(size*.5f);

		float depthColor =   1.0f -( pond.bounds.getNear()- position.z) * depthColorScale;

		gl.glPointSize(halfVisionRadius);
		gl.glBegin(GL.GL_POINTS);
		gl.glColor4f(depthColor,depthColor,1, .01f); 
		gl.glVertex2f(position.x, position.y);		
		gl.glEnd();


		gl.glBegin(GL.GL_TRIANGLES);
		gl.glColor3f(depthColor,depthColor,1); // z is between 0-1 0 is bottom
		gl.glVertex2f(position.x+drawHeading.x, position.y+drawHeading.y);
		gl.glVertex2f(position.x-drawHeading.x-drawHeading.y, position.y-drawHeading.y+drawHeading.x);
		gl.glVertex2f(position.x-drawHeading.x+drawHeading.y, position.y-drawHeading.y-drawHeading.x);
		gl.glEnd();


	}


	public float getSize() {
		return size;
	}


	public void setSize(float size) {
		this.size = size;
	}


	public boolean isVisibleToBoids() {
		return isVisibleToBoids;
	}


	public void setVisibleToBoids(boolean isVisibleToBoids) {
		this.isVisibleToBoids = isVisibleToBoids;
	}

	public void trigger(PolyRegion pr) {
		System.out.println(this + " is triggered");

	}


	public Vector3f getProposedPosition() {
		return proposedPosition;
	}


	public void setProposedPosition(Vector3f proposedPosition) {
		this.proposedPosition = proposedPosition;
	}


}
