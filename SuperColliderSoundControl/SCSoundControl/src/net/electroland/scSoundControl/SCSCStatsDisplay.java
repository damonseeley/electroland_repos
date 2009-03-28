package net.electroland.scSoundControl;

import java.util.Vector;

import processing.core.PApplet;
import processing.core.PFont;

public class SCSCStatsDisplay extends PApplet implements SCSoundControlNotifiable {

	boolean serverIsLive = false;
	private Vector<Float> avgCPUhistory;
	private Vector<Float> peakCPUhistory;
	private Vector<Float> polyphonyHistory;
	PFont theFont;
	int avgCpuColor = 0xff23B476;
	int peakCpuColor = 0xffA8B423;
	int polyphonyColor = 0xff237BB4;
	
	public void setup() {
		avgCPUhistory = new Vector<Float>();
		peakCPUhistory = new Vector<Float>();
		polyphonyHistory = new Vector<Float>();
		
		size(200, 200);
		noLoop();
		
		theFont = loadFont("ArialMT-16.vlw");
	}
	
	public void draw() {
		background(0);
		strokeWeight(1);
		
		if (!serverIsLive) {
			stroke(240, 10, 10);
			line(0, 0, width, height);
			line(0, height, width, 0);
		}
				
				
		//draw history graphs
		for (int i=2; i<width; i++) {
			stroke(avgCpuColor);
			if (avgCPUhistory.size() >= i) line(i-2, height * (1f - avgCPUhistory.get(i-2)), i-1, height * (1f - avgCPUhistory.get(i-1)));
			stroke(peakCpuColor);
			if (peakCPUhistory.size() >= i) line(i-2, height * (1f - peakCPUhistory.get(i-2)), i-1, height * (1f - peakCPUhistory.get(i-1)));
			stroke(polyphonyColor);
			if (polyphonyHistory.size() >= i) line(i-2, height * (1f - polyphonyHistory.get(i-2)), i-1, height * (1f - polyphonyHistory.get(i-1)));
		}
		
		//draw text
		stroke(250);
		textFont(theFont, 16);
		fill(avgCpuColor);
		if (avgCPUhistory.size() > 0) { text("% AverageCPU: " + nf(avgCPUhistory.get(0)*100f, 2, 1), 6, 20); }
		fill(peakCpuColor);
		if (peakCPUhistory.size() > 0) { text("% PeakCPU: " + nf(peakCPUhistory.get(0)*100f, 2, 1), 6, 40); }
		fill(polyphonyColor);
		if (polyphonyHistory.size() > 0) { text("% MaxPolyphony: " + nf(polyphonyHistory.get(0)*100f, 2, 1), 6, 60); }
	}
	

	//unneeded, but required by notifiable interface
	public void receiveNotification_BufferLoaded(int id, String filename) {}

	//go live
	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		redraw();
	}

	//get server load update
	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU) {
		avgCPUhistory.add(0, averageCPU/ 100f);
		if (avgCPUhistory.size() > width) avgCPUhistory.removeElementAt(avgCPUhistory.size()-1);
		peakCPUhistory.add(0, peakCPU/ 100f);
		if (peakCPUhistory.size() > width) peakCPUhistory.removeElementAt(peakCPUhistory.size()-1);
		redraw();
	}

	//not live
	public void receiveNotification_ServerStopped() {
		serverIsLive = false;		
		redraw();
	}
	
	public void notify_currentPolyphony(float polyphony) {
		polyphonyHistory.add(0, polyphony);
		if (polyphonyHistory.size() > width) polyphonyHistory.removeElementAt(polyphonyHistory.size() - 1);
	}
	
}
