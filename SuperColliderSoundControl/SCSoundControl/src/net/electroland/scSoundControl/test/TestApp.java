package net.electroland.scSoundControl.test;

import net.electroland.scSoundControl.*;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import processing.core.*;

public class TestApp extends PApplet {

	SCSoundControl ss;
	int counter1;
	int counter2;
	
	int soundSurge_period = 30; //period in seconds of a surge in number of sounds played
	int maxPolyphony = 32; // how many voices max to play at once
	
	int numSoundChannels = 2;
	
	int numSoundFiles = 12;
	String soundFilePath = "/Users/Jacob/Pro/ElectrolandWorkspace/SCSoundControl/testSounds/";
	String soundFilePrefix = "test_";
	String soundFileSuffix = ".wav";
	
	//buffer id's are stored in a list.
	//client app (e.g. this testApp) must keep track of which is which.
	Vector<Integer> _bufferList;
	Vector<SoundNode> _soundNodes;
	
	public void finalize() {
		ss.cleanup();
	}
	
	
	public void setup() {
		
		//internal inits
		_bufferList = new Vector<Integer>();
		_soundNodes = new Vector<SoundNode>();
		counter1 = counter2 = 11;
		
		//SoundControl inits
		ss = new SCSoundControl(numSoundChannels);
		ss.init();
		ss.showDebugOutput(false);
		
		initBuffers();

		//Processing inits
		size(200,200);
		frameRate(10);
		
	}
	
	
	public void draw() {
		
		//have a cycle 
		float timeFactor = (second() % soundSurge_period) / (float)soundSurge_period;
		float surgeFactor = 0.5f + 0.5f * sin(timeFactor*TWO_PI);

		background(255 * surgeFactor);

		int currentPolyphony = (int)random((maxPolyphony+1) * surgeFactor); //random is exclusive of its max, so add 1 to max polyphony.
		playToPolyphony(currentPolyphony);
		
		//cull dead sound nodes.
		cleanupSoundNodes();
	}
	
	//play random sound up to a max number of simultaneous voices 
	public void playToPolyphony(int targetPolyphony) {
		int curPolyphonyCount = 0;
		for (int i=0; i <_soundNodes.size(); i++) {
			if (_soundNodes.get(i).isAlive()) curPolyphonyCount++;
		}
		println("currentPolyphony: " + curPolyphonyCount);
		
		int newVoicesNeeded = targetPolyphony - curPolyphonyCount;
		if (newVoicesNeeded > 0) {
			for (int i = 0; i < newVoicesNeeded; i++) {
				_soundNodes.add(ss.createSoundNode(_bufferList.get((int)random(0,_bufferList.size())), false, generateRandomVolumeArray()));
			}
		}
		
	}
	
	//populate an array of floats with random values 0-1, size equal to number of output channels
	public float[] generateRandomVolumeArray() {
		float[] result = new float[numSoundChannels];
		for (int i = 0 ; i < numSoundChannels; i++) {
			result[i] = random(1.0f);
		}
		return result;
	}
	
	
	//go through the list of sound nodes. If one has finished playing, cull it from the list.
	public void cleanupSoundNodes() {
		SoundNode thisNode;
		Enumeration<SoundNode> e = _soundNodes.elements();
		while (e.hasMoreElements()) {
			thisNode = e.nextElement();
			if (!thisNode.isAlive()) _soundNodes.remove(thisNode);
		}
	}

	//just play one buffer on channels 0 and 1
	public void playBuffer(int bufferNumber) {
		if (_bufferList.contains(bufferNumber)) {
			_soundNodes.add(ss.createSoundNode(_bufferList.get(bufferNumber), false, new float[]{1f,1f}));
		} else {
			println("Buffer Number " + bufferNumber + " does not exist.");
		}
	}
	

	public void initBuffers() {
		//readBuf returns a buffer id. Keep track of it.
		//_bufferList.add(ss.readBuf(new String("sounds/a11wlk01.wav")));
		//_bufferList.add(ss.readBuf(new String("sounds/a11wlk01-44_1.aiff")));
		
		for (int i=1; i <= numSoundFiles; i++) {
			_bufferList.add(ss.readBuf(new String(soundFilePath + soundFilePrefix + i + soundFileSuffix)));
		}
	}
	
	
	static public void main(String args[]) {   PApplet.main(new String[] { "net.electroland.scSoundControl.test.TestApp" });}

}
