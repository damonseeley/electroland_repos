package net.electroland.detector;

import java.awt.Image;
import java.util.List;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;

import processing.core.PImage;

/**
 * This is an abstract class that acts as a bridge between detectors, rasters,
 * and DMX lighting fixtures.  The only method that concrete implementations
 * need to implement is send(byte[]), which actually sends the bytes to the
 * fixture, using the appropriate protocol.  See ArtNetDMXLightingFixture.
 * @author geilfuss
 */
public abstract class DMXLightingFixture {

	protected byte universe;
	protected InetAddress ip;
	protected String ipStr, id;
	protected int port, channels;
	protected List <Detector> detectors;
	protected int width, height; // for generating raster properly.
	protected String lightgroup;
	//protected String color;	// physical color of fixture
	//protected int soundChannel; //sound channel to play on for this fixture.

	/**
	 * @param universe - the byte id of this lighting fixtures DMX universe
	 * @param ipStr - the ip address for the lighting controller for this fixture
	 * @param port - the port the lighting controller is listening on
	 * @param channels - the total number light channels this fixture is supporting
	 * @throws UnknownHostException
	 */
	public DMXLightingFixture(String id, byte universe, String ipStr, int port, int channels, int width, int height) throws UnknownHostException {
		this.universe = universe;
		this.ipStr = ipStr;
		this.port = port;

		this.width = width;
		this.height = height;
		this.id = id;
		this.setChannels(channels);
		
		this.ip = InetAddress.getByName(ipStr);
		this.detectors = Collections.synchronizedList(new ArrayList<Detector>(channels));
		this.channels = channels;
	}
	
	/**
	 * Implement this.
	 * 
	 * @param data
	 */
	abstract void send(byte[] data);
	
	/**
	 * Sets the total number of channels for this fixure.  This method preserve
	 * your existing channel detectors, but it will discard any that are
	 * addressed beyond the number of channesl you are setting to.
	 * 
	 * @param channels
	 */
	final public void setChannels(int channels) {
		this.channels = channels;		
	}
	
	final public void setChannelDetector(int channel, Detector detector) throws ArrayIndexOutOfBoundsException, NullPointerException{
		detectors.add(channel, detector);
	}

	/**
	 * Whoops.  this belongs in Fixture.
	 * 
	 * @param PImage
	 */
	final public void sync(PImage raster){
		byte[] data = new byte[channels];
		for (int i = 0; i < data.length; i++){
			Detector detector = detectors.get(i);
			if (detector == null){
				data[i] = 0;
			}else{
				// populate the pixel buffer
				// (if we need to optimize later, I can almost guarantee we
				//  should start here, by doing direct System.arrayCopy operations
				//  from raster.pixels.)
				PImage subraster = raster.get(detector.x, 
												detector.y, 
												detector.width, 
												detector.height);
				subraster.updatePixels();
				subraster.loadPixels();
				data[i] = (byte)detector.model.getValue(subraster.pixels);
			}
		}
		send(data);
	}

	/**
	 * NOT IMPLEMENTED YET.
	 * @param image
	 */
	final public void sync(Image raster){
		// to do: IMPLEMENT
	}

	public int getChannels() {
		return channels;
	}
	
	public List<Detector> getDetectors(){
		return detectors;
	}
	
	final public byte getUniverse() {
		return universe;
	}
	
	final public InetAddress getIp() {
		return ip;
	}

	final public String getIpStr() {
		return ipStr;
	}

	final public int getPort() {
		return port;
	}
	
	final public String getID() {
		return id;
	}
	
	/*
	final public String getColor(){
		return color;
	}
	
	final public int getSoundChannel(){
		return soundChannel;
	}
	*/
}