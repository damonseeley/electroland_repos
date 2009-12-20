package net.electroland.scSoundControl.test;
import net.electroland.scSoundControl.*;
import processing.core.*;

public class SimpleTest extends PApplet implements SCSoundControlNotifiable {

	SCSoundControl ss;
	
	//test behavior parameters
	int soundSurge_period = 10; //period in seconds of a surge in number of sounds played
	
	//how many output channels. If there are more than 8 inputs, we'll need to add a variable for that too.
	int numOutputChannels = 2; 
	int numInputChannels = 2;
	
	//test parameters
	String soundFile1, soundFile2; 
	
	SoundNode sound1, sound2;
	int buffer1, buffer2;
	
	boolean serverIsLive = false;
	
	PFont theFont;
	
	public void setup() {		

		soundFile1 = sketchPath("soundfiles/StereoTest.wav");
		//soundFile1 = sketchPath("soundfiles/test_1.wav");
		//soundFile2 = sketchPath("soundfiles/test_11.wav");

		//use an init method, since processing is calling setup twice (a workaround)
		initTestApp();
		
		//Processing inits
		size(400,400);
		frameRate(30);
	}
	
	private void loadBuffers() {
		ss.readBuf(soundFile1);
		//ss.readBuf(soundFile2);
	}
	
	//processing core is for some reason calling setup twice! Don't know why that's
	//happening, and since this is just test code we'll work around it for now.
	boolean isInitialized = false;
	public void initTestApp() {
		
		//only initialize once.
		if (isInitialized) return;
		
		//SoundControl inits
		ss = new SCSoundControl(this);
		
		//ss.showDebugOutput(false);
		ss.showDebugOutput(true);

		theFont = loadFont("ArialMT-16.vlw");

		isInitialized = true;
	}
	
	float howMuchX = 1, howMuchY = 1;
	
	public void draw() {
		background(0);
		strokeWeight(1);
		
		if (!serverIsLive) {
			stroke(240, 10, 10);
			line(0, 0, width, height);
			line(0, height, width, 0);
		}
		
		//have a cycle 
		float timeFactor = (second() % soundSurge_period) / (float)soundSurge_period;
		//float surgeFactor = 0.5f + 0.5f * sin(timeFactor*TWO_PI);

		if (mousePressed) howMuchY = (height - mouseY) / (float)height;
		if (mousePressed) howMuchX = mouseX / (float)width;
		
		if (sound1 != null) {
			if (sound1.isAlive()) sound1.setAmplitude(0, 0, howMuchY);
			if (sound1.isAlive()) sound1.setAmplitude(0, 1, 1f-howMuchY);
			if (sound1.isAlive()) sound1.setAmplitude(1, 0, howMuchX);
			if (sound1.isAlive()) sound1.setAmplitude(1, 1, 1f-howMuchX);
		}

	}
	
	public void keyPressed() {
		if (key == 'q') {
			if (ss != null) ss.shutdown();
			System.exit(0);
		}
		if (key == '1') {
		}
		else if (key == '2') {
		}
	}
	
	public void mousePressed() {
		if (serverIsLive) sound1 = ss.createStereoSoundNode(buffer2, true, new int[]{0, 1}, new float[]{1, 1}, new int[]{0, 1}, new float[]{1, 1}, 0.85f);
	}
	
	public void mouseReleased() {
		if (sound1 != null) sound1.die();
	}

	public void receiveNotification_ServerRunning() {
		println("Server running.");
		serverIsLive = true;
		loadBuffers();
	}


	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
		println("Server stopped.");
	}

	public void receiveNotification_ServerStatus(float averageCPUload, float peakCPUload, int numSynths) {
		//simple test doesn't care how hard the computer is working.
	}
	
	public void receiveNotification_BufferLoaded(int id, String filename) {
		println("Loaded buffer " + id + ", " + filename);
		if (filename.compareToIgnoreCase(soundFile1) == 0) buffer1 = id;
		else if (filename.compareToIgnoreCase(soundFile2) == 0) buffer2 = id;
	}

	
	static public void main(String args[]) {   PApplet.main(new String[] { "net.electroland.scSoundControl.test.SimpleTest" });}


}
