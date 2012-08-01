package net.electroland.fish.core;

import java.awt.Polygon;
import java.awt.event.ComponentListener;
import java.awt.event.FocusListener;
import java.awt.event.HierarchyBoundsListener;
import java.awt.event.HierarchyListener;
import java.awt.event.InputMethodListener;
import java.awt.event.KeyListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelListener;
import java.beans.PropertyChangeListener;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLContext;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLException;
import javax.vecmath.Vector3f;

import net.electroland.broadcast.fish.PoolXMLSocketMessage;
import net.electroland.broadcast.server.XMLSocketBroadcaster;
import net.electroland.fish.boids.InvisibleSoundFish;
import net.electroland.fish.forces.Wave;
import net.electroland.fish.ui.Drawable;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;

import com.sun.opengl.util.FPSAnimator;



public class PondWTouchLimit extends Thread implements Drawable {

	public PondGenerator generator;

	public Vector<Boid> offScreen = new Vector<Boid>();

	public ContentLayout contentLayout;

	public static  boolean USE_TIMER = FishProps.THE_FISH_PROPS.getProperty("useTimer", false);

	public static final boolean traceXML = false;

	public FrameRateCalculator frameRateCalc;
	public FrameRateCalculator xmlFrameRateCalc;
	public boolean isRunning;

//	Timer timer;

	public Bounds bounds;
	public Bounds wallAvoidBounds;
	public Bounds centerIslandBounds;

	public static long pondPopulationDelay = FishProps.THE_FISH_PROPS.getProperty("pondPopulationDelay", 250);
	public long pondPopulationTime = 0;

	float buffersize = 100;
	float zBuffersize = .01f;

	public static int visionDistanceSqr = 1000;

	public ConcurrentHashMap<Boid, Boid> boids = new ConcurrentHashMap<Boid, Boid>();

	public SpacialGrid grid;

	XMLSocketBroadcaster xmlsb;
	PoolXMLSocketMessage mssg;

	long intendedDelay;
	long timeToNextSend;

	long eventTime;
	boolean ambientPlayed = false;

	Polygon testPoly = new Polygon();


	@SuppressWarnings("unchecked") // needed for spacialGrid
	public PondWTouchLimit(Bounds bounds, int xCellCnt, int yCellCnt) { // cell size should be 1/2 smallest view radius of boid
		System.out.println("pond bounds: " + bounds);
		this.bounds = new Bounds(bounds);

		contentLayout = new ContentLayout(bounds);


		grid = new SpacialGrid(bounds,xCellCnt, yCellCnt);
		int intendedFrameRate = FishProps.THE_FISH_PROPS.getProperty("pondFrameRate", 50);
		intendedDelay = (long) (1000.0/(double)intendedFrameRate);
		System.out.println("intended frame rate:" +  intendedFrameRate + "    indended delay:" + intendedDelay);
		timeToNextSend = intendedDelay;

//		timer = new Timer(intendedFrameRate);
//		timer = new Timer();
		frameRateCalc = new FrameRateCalculator(intendedFrameRate * 5, intendedFrameRate); // 5 second ave
		xmlFrameRateCalc = new FrameRateCalculator(intendedFrameRate * 5, intendedFrameRate); // 5 second ave


//		timer.scheduleAtFixedRate(new RunnerTask(), 0,  (long)frameRateCalc.elapsedTime);


		wallAvoidBounds = new Bounds(
				bounds.getTop() + buffersize,
				bounds.getLeft() + buffersize,
				bounds.getBottom() - buffersize,
				bounds.getRight() - buffersize,
				bounds.getNear() - zBuffersize,
				bounds.getFar() + zBuffersize);

		centerIslandBounds = new Bounds(576,896, 960, 2176, bounds.getNear(),bounds.getFar());

		xmlsb = new XMLSocketBroadcaster(FishProps.THE_FISH_PROPS.getProperty("flexPort", 1024));
		xmlsb.start();

		mssg = new PoolXMLSocketMessage(Boid.broadcastFishList);



		eventTime = System.currentTimeMillis() + 1000 * 10; // 10 secs
	}

	public void startRendering() {
		if(USE_TIMER) {
			new  FPSAnimator(new RunnerTask(), 40).start();
		} else {
			start();			
		}
	}


	public void add(Boid b) {
		if(b.offScreen) {
			offScreen.add(b);
		} else {
			boids.put(b, b); // will add self to grid next frame
		}
	}

	public void remove(Boid b) {
		boids.remove(b);
		grid.cells[b.scpacialGridLoc.x][b.scpacialGridLoc.y].remove(b);

	}






	public  long CUR_TIME = 0;
	public  float ELAPSED_TIME = 0;
	public static final float SEC_TO_MIL = 1000f;
	public static  final float  MIL_TO_SEC = 1.0f / SEC_TO_MIL;
	public  float CURFRAME_TIME_SCALE;


	public void startWave() {
		Wave.waveScalerInc = 0;
		Wave.waveState = Wave.WaveState.active;
		eventTime = CUR_TIME + FishProps.THE_FISH_PROPS.getProperty("waveDuration", 15000);
//		Wave.waveStartTime = CUR_TIME;
		InvisibleSoundFish.playWaveSound();

	}

	public void stopWave() {
		Wave.waveState = Wave.WaveState.inactive;
		eventTime = CUR_TIME + FishProps.THE_FISH_PROPS.getProperty("waveDelay", 1000 * 60 * 5);
	}

	public void toggleWave() {
		if(Wave.waveState == Wave.WaveState.inactive) {
			startWave();
		} else {
			stopWave();
		}
	}

	float waveRame = FishProps.THE_FISH_PROPS.getProperty("waveRamp", .01f);

	
	public static final int touchLimit =  FishProps.THE_FISH_PROPS.getProperty("touchLimit", 1000);

	public void runFrame() {

		if(CUR_TIME > eventTime) {
			if(ambientPlayed) {
				toggleWave();
			} else {
				System.out.println("playing ambient");
				InvisibleSoundFish.playAmbientSound();
				eventTime = CUR_TIME + FishProps.THE_FISH_PROPS.getProperty("waveDelay", 1000 * 60 * 5);
				ambientPlayed = true;
			}

		} else if (Wave.waveState == Wave.WaveState.active) {
			Wave.waveScalerInc += .001;
//			Wave.waveElapsedTime = CUR_TIME - Wave.waveStartTime;
		}

		if(pondPopulationTime < CUR_TIME) {
			if(! offScreen.isEmpty()) {
				Boid b = offScreen.remove((int) Math.floor(Math.random() * offScreen.size()));	
				b.offScreen = false;
				add(b);
				generator.setStartPosition(b, bounds);
			
			}
			pondPopulationTime = CUR_TIME + pondPopulationDelay;
		}

		Vector<Boid> seenBoids = new Vector<Boid>();

		
		int touchsLeft = touchLimit;


		
		// figure out change in velociy (but don't move)
		Enumeration<Boid> e = boids.elements();
		while(e.hasMoreElements()) {
			Boid b = e.nextElement();
			seenBoids = grid.getBoidsInRadius(b.position, b.visionRadius);
			b.reactToBoids(seenBoids);
			if(touchsLeft >= 0) {
				b.senceTouch();
				if(b.isTouched) {
					touchsLeft--;
				}
			}
			b.applyForces();
		}

		e = boids.elements();
		while(e.hasMoreElements()) {
			Boid b = e.nextElement();
			b.move();

//			if(b instanceof VideoFish) {
//			System.out.println("o:" + b.broadcastFish.orientation);
//			}
		}

		frameRateCalc.frame();
		CUR_TIME = frameRateCalc.curTime;
		ELAPSED_TIME =  frameRateCalc.elapsedTime;
		CURFRAME_TIME_SCALE = MIL_TO_SEC * ELAPSED_TIME;





//		intendedDelay = (long) (1.0/(double)intendedFrameRate);
//		timeToNextSend = intendedDelay;


		//	timer.block();



	}

	public void run() {

		if(USE_TIMER) {
			System.out.println("run was called with use timer one this shouldn't happen");
		} else {
			isRunning = true;

			while(isRunning) {
				runFrame();

				if(timeToNextSend < CUR_TIME) {
					xmlFrameRateCalc.frame(frameRateCalc.curTime);
					if(traceXML) {
						System.out.println(mssg.toXML());
					}
					xmlsb.send(mssg);
					timeToNextSend = CUR_TIME + intendedDelay;
				} 
			}

		}
	}
	public class RunnerTask implements GLAutoDrawable {


		public void addGLEventListener(GLEventListener arg0) {
		}

		public void display() {
			runFrame();
			if(traceXML) {
				System.out.println(mssg.toXML());
			}
			xmlsb.send(mssg);
		}

		public boolean getAutoSwapBufferMode() {
			return false;
		}

		public GLContext getContext() {
			return null;
		}

		public GL getGL() {
			return null;
		}

		public void removeGLEventListener(GLEventListener arg0) {			
		}

		public void repaint() {
		}

		public void setAutoSwapBufferMode(boolean arg0) {
		}

		public void setGL(GL arg0) {
		}

		public GLContext createContext(GLContext arg0) {
			return null;
		}

		public int getHeight() {
			return 0;
		}

		public int getWidth() {
			return 0;
		}

		public void setRealized(boolean arg0) {

		}

		public void setSize(int arg0, int arg1) {

		}

		public void swapBuffers() throws GLException {

		}

		public void addComponentListener(ComponentListener arg0) {

		}

		public void addFocusListener(FocusListener arg0) {

		}

		public void addHierarchyBoundsListener(HierarchyBoundsListener arg0) {

		}

		public void addHierarchyListener(HierarchyListener arg0) {

		}

		public void addInputMethodListener(InputMethodListener arg0) {

		}

		public void addKeyListener(KeyListener arg0) {

		}

		public void addMouseListener(MouseListener arg0) {

		}

		public void addMouseMotionListener(MouseMotionListener arg0) {			
		}

		public void addMouseWheelListener(MouseWheelListener arg0) {

		}

		public void addPropertyChangeListener(PropertyChangeListener arg0) {

		}

		public void addPropertyChangeListener(String arg0,
				PropertyChangeListener arg1) {

		}

		public void removeComponentListener(ComponentListener arg0) {

		}

		public void removeFocusListener(FocusListener arg0) {

		}

		public void removeHierarchyBoundsListener(HierarchyBoundsListener arg0) {

		}

		public void removeHierarchyListener(HierarchyListener arg0) {
		}

		public void removeInputMethodListener(InputMethodListener arg0) {
		}

		public void removeKeyListener(KeyListener arg0) {
		}

		public void removeMouseListener(MouseListener arg0) {
		}

		public void removeMouseMotionListener(MouseMotionListener arg0) {
		}

		public void removeMouseWheelListener(MouseWheelListener arg0) {
		}

		public void removePropertyChangeListener(PropertyChangeListener arg0) {
		}

		public void removePropertyChangeListener(String arg0,
				PropertyChangeListener arg1) {

		}

		public GLCapabilities getChosenGLCapabilities() {
			return null;
		}

	}

	public void draw(GLAutoDrawable drawable, GL gl) {
//		gl.glColor3f(.03f, .03f, .03f);
//		gl.glRectf(wallAvoidBounds.getLeft(), wallAvoidBounds.getTop(), wallAvoidBounds.getRight(), wallAvoidBounds.getBottom());

//		gl.glColor3f(.06f, .05f, .05f);
//		gl.glRectf(centerIslandBounds.getLeft(), centerIslandBounds.getTop(),centerIslandBounds.getRight(), centerIslandBounds.getBottom());



		grid.draw(drawable, gl);


		Enumeration<Boid> e = boids.elements();
		while(e.hasMoreElements()) {
			e.nextElement().draw(drawable, gl);

		}




	}


}
