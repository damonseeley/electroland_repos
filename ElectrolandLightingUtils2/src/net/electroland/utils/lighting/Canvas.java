package net.electroland.utils.lighting;

import java.util.Hashtable;
import java.util.Vector;

abstract class Canvas {

	private String name;

	abstract Object getSurface();

	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}	

	abstract Hashtable<Fixture,Vector <Detector>> syncDetectorStates();
}
