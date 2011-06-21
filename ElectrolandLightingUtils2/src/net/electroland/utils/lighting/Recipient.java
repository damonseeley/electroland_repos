package net.electroland.utils.lighting;

import java.util.Hashtable;
import java.util.Vector;

abstract class Recipient {
	
	private String name;
	private Vector<Fixture> fixtures;
	private int channels;


	public void setName(String name){
		this.name = name;
	}
	
	public String getName(){
		return name;
	}

	// send data from all fixtures
	public void sync(Hashtable<Fixture,Vector<Detector>> detectorStates){
		// cycle through all fixtures
		// get the values from each detector, and plop them into the byte array
		// send(bytearray);
	}

	// send all "on" values
	abstract public void allOn();
	
	// send all "off" values
	abstract public void allOff();

	abstract public void send(byte[] b);
}