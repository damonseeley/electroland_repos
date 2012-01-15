package net.electroland.scSoundControl.test;

import java.util.Enumeration;
import java.util.Vector;
import processing.core.*;
import net.electroland.scSoundControl.*;

@SuppressWarnings("serial")
public class AutomatedTestApp extends PApplet implements SCSoundControlNotifiable {

	SCSoundControl ss;
	
	//test behavior parameters
	int soundSurge_period = 5; //period in seconds of a surge in number of sounds played
	int maxPolyphony = 64; // how many voices max to play at once
	
	// polyphony values for automated switching
	int polyphonyA = 4;
	int polyphonyB = 8;
	int polyphonyC = 16;
	int polyphonyD = 24;
	int polyphonyMode = 0;
	long lastPolyphonySwitch;
	int polyphonyDuration = 10000;
	int targetPolyphony;
	
	//how many output channels. If there are more than 8 inputs, we'll need to add a variable for that too.
	//this is sort of hacky, should really all be in the props file
	int numOutputChannels = 22; 
	int numInputChannels = 2;
	
	//test parameters
	int numSoundFiles = 12;
	String soundFilePath = "soundfiles/44.1/"; //path relative to processing sketch folder
	String soundFilePrefix = "test_";
	String soundFileSuffix = ".wav";
	
	//buffer id's are stored in a list.
	//client app (e.g. this testApp) must keep track of which is which.
	Vector<Integer> _bufferList;
	Vector<SoundNode> _soundNodes;
	
	boolean serverIsLive = false;
	
	PFont theFont;
	
	float avgCPU, peakCPU;
	
	Vector<Integer> polyphonyHistory;
	Vector<Integer> avgCPUhistory;
	Vector<Integer> peakCPUhistory;
	
	int polyphonyColor = 0xff237BB4;
	int avgCpuColor = 0xff23B476;
	int peakCpuColor = 0xffA8B423;
	
	//update sound data on a longer interval than frame udpates:
	int soundUpdateInterval = 100; //in millis
	long lastSoundUpdateTime = 0;
	
	//this should get called when things are shutting down. It often doesn't.
	public void finalize() {
		ss.cleanup();
	}
	
	
	public void setup() {		
				
		//use an init method, since processing is calling setup twice (a workaround)
		initTestApp();

		//Processing inits
		size(400,400);
		frameRate(10);
		
	}
	
	//processing core is for some reason calling setup twice! Don't know why that's
	//happening, and since this is just test code we'll work around it for now.
	boolean isInitialized = false;
	public void initTestApp() {
		
		//only initialize once.
		if (isInitialized) return;
		
		soundFilePath = sketchPath(soundFilePath);
		//double check the soundfile path:
		println("Looking for sound files in " + soundFilePath);

		//internal inits
		_bufferList = new Vector<Integer>();
		_soundNodes = new Vector<SoundNode>();
		polyphonyHistory = new Vector<Integer>();
		peakCPUhistory = new Vector<Integer>();
		avgCPUhistory = new Vector<Integer>();
		
		//SoundControl inits
		ss = new SCSoundControl(this, "depends/SCSoundControl.properties");
		ss.init();
		
		ss.showDebugOutput(false);
		//ss.showDebugOutput(true);

		theFont = loadFont("ArialMT-16.vlw");

		isInitialized = true;
		
		// start timer
		lastPolyphonySwitch = System.currentTimeMillis();
		targetPolyphony = polyphonyA;
	}
	
	//a unique function to reinitialize the test app.
	public void reInit() {
		isInitialized = false;
		initTestApp();
	}

	public void loadBuffers() {		
		//keep track of buffer numbers returned.
		for (int i=1; i <= numSoundFiles; i++) {
			String bufferName = new String(soundFilePath + soundFilePrefix + i + soundFileSuffix);
			//_bufferList.add(ss.readBuf(bufferName));
			ss.readBuf(bufferName);
			println("Requested buffer " + bufferName);
		}		
	}
	
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
		float surgeFactor = 0.5f + 0.5f * sin(timeFactor*TWO_PI);

		//targetPolyphony = (int)random((maxPolyphony+1) * surgeFactor); //random is exclusive of its max, so add 1 to max polyphony.

		//if pressing space, take polyphony from mouseY (UP is more)
		//if (keyPressed && key == ' ') currentPolyphony = (int)(maxPolyphony * (height - mouseY) / (float)height);
		//if (mousePressed) targetPolyphony = (int)(maxPolyphony * (height - mouseY) / (float)height);
		
				
		//draw history graphs
//		for (int i=2; i<width; i++) {
//			stroke(polyphonyColor);
//			if (polyphonyHistory.size() >= i) line(i-2, polyphonyHistory.get(i-2), i-1, polyphonyHistory.get(i-1));
//			stroke(avgCpuColor);
//			if (avgCPUhistory.size() >= i) line(i-2, avgCPUhistory.get(i-2), i-1, avgCPUhistory.get(i-1));
//			stroke(peakCpuColor);
//			if (peakCPUhistory.size() >= i) line(i-2, peakCPUhistory.get(i-2), i-1, peakCPUhistory.get(i-1));
//		}
		
		//draw text
		stroke(250);
		textFont(theFont, 16);
		fill(polyphonyColor);
		text("TargetPolyphony (cur): " + targetPolyphony + " (" + getCurPolyphony() + ")", 6, 20);
		fill(avgCpuColor);
		text("% AverageCPU: " + nf(avgCPU, 2, 1), 6, 40);
		fill(peakCpuColor);
		text("% PeakCPU: " + nf(peakCPU, 2, 1), 6, 60);

		//oscillate the playback rates
		for (int i=0; i <_soundNodes.size(); i++) {
			// these objects are occasionally null. not sure why, but best to check for it.
			if(_soundNodes.get(i) != null){
				if (_soundNodes.get(i).isAlive()) _soundNodes.get(i).setPlaybackRate(surgeFactor);
			}
		}

		
		//somewhate independently from the frame rate, update the sound playback
		if (millis() - lastSoundUpdateTime > soundUpdateInterval) {
			playToPolyphony(targetPolyphony);

			//cull dead sound nodes.
			cleanupSoundNodes();
			lastSoundUpdateTime = millis();
		}
		
		if(System.currentTimeMillis() - lastPolyphonySwitch > polyphonyDuration){
			if(polyphonyMode == 0){
				targetPolyphony = polyphonyA;
				polyphonyMode = 1;
			} else if(polyphonyMode == 1){
				targetPolyphony = polyphonyB;
				polyphonyMode = 2;
			} else if(polyphonyMode == 2){
				targetPolyphony = polyphonyC;
				polyphonyMode = 3;
			} else if(polyphonyMode == 3){
				targetPolyphony = polyphonyD;
				polyphonyMode = 0;
			}
			System.out.println("Current polyphony: "+ targetPolyphony);
			//playToPolyphony(targetPolyphony);
			//cleanupSoundNodes();
			lastPolyphonySwitch = System.currentTimeMillis();
		}

	}
	
	public void keyPressed() {
		if (key == 'q') {
			ss.shutdown();
			System.exit(0);
		}
	}
	
	public int getCurPolyphony() {
		int curPolyphonyCount = 0;
		for (int i=0; i <_soundNodes.size(); i++) {
			// these objects are occasionally null. not sure why, but best to check for it.
			if(_soundNodes.get(i) != null){
				if (_soundNodes.get(i).isAlive()) curPolyphonyCount++;
			}
		}
		return curPolyphonyCount;
	}
	
	//play random sound up to a max number of simultaneous voices
	//returns the previous polyphony
	public void playToPolyphony(int targetPolyphony) {
		if (!serverIsLive || _bufferList.size() < 1) return;

		int newVoicesNeeded = targetPolyphony - getCurPolyphony();
		if (newVoicesNeeded > 0) {
			for (int i = 0; i < newVoicesNeeded; i++) {
				_soundNodes.add(ss.createMonoSoundNode(_bufferList.get((int)random(0,_bufferList.size())), false, generateOutputChannelArray(), generateRandomVolumeArray(), 2f));
			}
		}
	}
	
	//populate an array of floats with random values 0-1, size equal to number of output channels
	public float[] generateRandomVolumeArray() {
		float[] result = new float[numOutputChannels];
		for (int i = 0 ; i < numOutputChannels; i++) {
			result[i] = random(1.0f);
		}
		return result;
	}
	
	//populate an array of floats with random values 0-1, size equal to number of output channels
	public int[] generateOutputChannelArray() {
		int[] result = new int[numOutputChannels];
		for (int i = 0 ; i < numOutputChannels; i++) {
			result[i] = i;
		}
		return result;
	}
	
	//go through the list of sound nodes. If one has finished playing, cull it from the list.
	protected void cleanupSoundNodes() {
		SoundNode thisNode;
		Enumeration<SoundNode> e = _soundNodes.elements();
		while (e.hasMoreElements()) {
			thisNode = e.nextElement();
			if(thisNode != null){
				if (!thisNode.isAlive()) _soundNodes.remove(thisNode);
			}
		}
	}

	//just play one buffer on channels 0 and 1
	public void playBuffer(int bufferNumber) {
		if (!serverIsLive) return;
		
		if (_bufferList.contains(bufferNumber)) {
			SoundNode thisNode = ss.createMonoSoundNode(_bufferList.get(bufferNumber), false, new int[]{0, 1}, new float[]{1f,1f}, 2f);
			if (thisNode != null) { _soundNodes.add(thisNode); }
			else { println("Unable to create new sound node for buffer " + bufferNumber); }
		} else {
			println("Buffer Number " + bufferNumber + " does not exist.");
		}
	}

	public void receiveNotification_ServerRunning() {
		
		println("Server running.");
		_bufferList = new Vector<Integer>();
		_soundNodes = new Vector<SoundNode>();
		
		this.loadBuffers();

		serverIsLive = true;
	}


	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
		println("Server stopped.");
	}

	public void receiveNotification_ServerStatus(float averageCPUload, float peakCPUload, int numSynths) {
		avgCPU = averageCPUload;
		peakCPU = peakCPUload;

		//push the latest data points into graph history, and remove old ones
		polyphonyHistory.add(0, (int)((1f-(getCurPolyphony()/(float)maxPolyphony)) * height));
		if (polyphonyHistory.size() > width) polyphonyHistory.removeElementAt(polyphonyHistory.size()-1);
		avgCPUhistory.add(0, floor(height * (1f - (avgCPU/ 100f))));
		if (avgCPUhistory.size() > width) avgCPUhistory.removeElementAt(avgCPUhistory.size()-1);
		peakCPUhistory.add(0, floor(height * (1f - (peakCPU/ 100f))));
		if (peakCPUhistory.size() > width) peakCPUhistory.removeElementAt(peakCPUhistory.size()-1);
	}
	
	public void receiveNotification_BufferLoaded(int id, String filename) {
		println("Loaded buffer " + id + ", " + filename);
		_bufferList.add(id);
	}

	
	static public void main(String args[]) {
		PApplet.main(new String[] { "net.electroland.scSoundControl.test.AutomatedTestApp" });
	}



}

