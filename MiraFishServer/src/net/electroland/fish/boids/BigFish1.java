package net.electroland.fish.boids;

import java.util.Vector;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.vecmath.Vector3f;

import net.electroland.fish.core.Pond;
import net.electroland.fish.core.ContentLayout.Slot;
import net.electroland.fish.forces.BigFishEdgeBuffer;
import net.electroland.fish.forces.DiveOnMedia;
import net.electroland.fish.forces.Friction;
import net.electroland.fish.forces.MoveToPointAndPlayVid;
import net.electroland.fish.util.ContentList;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;

import com.sun.opengl.util.GLUT;

public class BigFish1 extends StandardFish {
	public static Vector<BigFish1> THEBIGFISH = new Vector<BigFish1>();
	//public Content content;
	
	
	
	String keyIndex = "";
	
	public void setIndex(int keyIndex ) {
		this.keyIndex = Integer.toString(keyIndex);
		
	}
	public enum State { SWIMING, VIDEO, DIVE }
	
	public State state = State.SWIMING;
	
	//public void setContent(Content c) {
		//content = c;
		
		//System.out.println("key " + (THEBIGFISH.indexOf(this) + 1) + "  is content " + c.name);
	//}
//	FishPolyRegion fpr;
	public BigFish1(Pond pond) {
		super(pond, 1f, 0);
		THEBIGFISH.add(this);
		this.pond = pond;
		this.species = FishIDs.BIGFISH1_ID;
		this.setSize(128);
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));


		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));
		
		
		setVelocity(new Vector3f(10 - 20*(float)Math.random(),10 - 20*(float)Math.random(),0));
		
		

		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("BigFish1MaxSpeed", 50), 10));


		//forces
		

		isBigFish = true;

		
		add("friction", new Friction(1f, this, .0001f));

		

		
		// second speed is move to show vid
		add("moveToPointAndPlayVid", new MoveToPointAndPlayVid(5f, this, 400f));
	
		add("diveOnMedia", new DiveOnMedia(10f, this, FishProps.THE_FISH_PROPS.getProperty("BigFishDiveSpeed", .05f)));
		
		//add("randomTurn", new RandomTurn(2f, this, 10000, 5000, 500, .5f));
		
		add("bigFishEdgeBuffer", new BigFishEdgeBuffer(2f, this, Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("BigFish1EdgeBufferSize", 300), 25f), FishProps.THE_FISH_PROPS.getProperty("BigFish1EdgeBufferStrength", 1f)));

		pond.add(this);

	}
	

	public void touched() {
		mediaTriggerAutoTimer = pond.CUR_TIME+ mediaTriggerPeriod;
		if(state == State.SWIMING) {
			super.touched(); // triggers accent
			state = State.VIDEO;
			moveToCornerAndPlayVideo();
		}
	}
	
	
	long mediaTriggerPeriod =  FishProps.THE_FISH_PROPS.getProperty("mediaTriggerPeriod", 1000 * 60 * 5);
	long mediaTriggerAutoTimer = System.currentTimeMillis() + ((long) (Math.random() * (float) mediaTriggerPeriod)) + 1000;
	
	public void move() {
		super.move();
		if(mediaTriggerAutoTimer < pond.CUR_TIME) {
			System.out.println("auto timer");
			touched();
		}
	}
/*	
	public static void generate(Pond p) throws IOException {
		HashMap<String, Content> cl = ContentList.getTheContactList();
		
		for(Content c : cl.values()) {
			BigFish1 bf = new BigFish1(p);
			bf.setContent(c);
		}
	}
*/
	
	public void reset() {
		state =State.SWIMING;
		MoveToPointAndPlayVid mtp = (MoveToPointAndPlayVid) getForce("moveToPointAndPlayVid");
		mtp.setGoal(null, 1, 1);
	
	}
	public void moveToCornerAndPlayVideo() {
		Slot slot = pond.contentLayout.getFreeSlot(position);
		
		if (slot == null) {
	//		System.out.println("slot filled can't diplay video");
			//DS ONSITE CHANGE
			state = State.DIVE;
			return;
		} else {
			VideoFish video =	new VideoFish(pond, 1f, FishIDs.VIDEOFISH_ID, ContentList.next());
			slot.vid = video;
			MoveToPointAndPlayVid mtp = (MoveToPointAndPlayVid) getForce("moveToPointAndPlayVid");
			mtp.setVideoFish(video);
			mtp.enable(true, slot);
		}
		

	}
	
	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			new BigFish1(p);
		}
	}

    int font = GLUT.BITMAP_HELVETICA_10;
    public final GLUT glut = new GLUT();
	public void draw(GLAutoDrawable drawable, GL gl) {
		super.draw(drawable, gl);
		gl.glColor3f(1,0,0);

		gl.glRasterPos3f(position.x, position.y, 0);
		glut.glutBitmapString(font, keyIndex);
        
	}
	
}