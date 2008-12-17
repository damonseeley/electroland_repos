package net.electroland.detector;

import java.awt.Image;
import java.net.InetAddress;
import java.net.UnknownHostException;

import processing.core.PImage;

/**
 * This is an abstract class that acts as a bridge between detectrors, rasters,
 * and DMX lighting fixtures.  The only method that concrete implementations
 * need to implement is send(byte[]), which actuall sends the bytes to the
 * fixture, using the appropriate protocol.  See ArtNetDMXLightingFixture.
 * @author geilfuss
 */
public abstract class DMXLightingFixture {

	protected byte universe;
	protected InetAddress ip;
	protected String ipStr, id;
	protected int port, channels;
	protected Detector[] detectors;
	protected int width, height; // for generating raster properly.
	protected String lightgroup;

	/**
	 * @param universe - the byte id of this lighting fixtures DMX universe
	 * @param ipStr - the ip address for the lighting controller for this fixture
	 * @param port - the port the lighting controller is listening on
	 * @param channels - the total number light channels this fixture is supporting
	 * @throws UnknownHostException
	 */
	public DMXLightingFixture(byte universe, String ipStr, int port, int channels, int width, int height, String id) throws UnknownHostException {
		this.universe = universe;
		this.ipStr = ipStr;
		this.port = port;

		this.width = width;
		this.height = height;
		this.id = id;
		this.setChannels(channels);
		
		this.ip = InetAddress.getByName(ipStr);
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
			if (detectors != null){
				synchronized (detectors){
					Detector[] newArray = new Detector[channels];
					System.arraycopy(detectors,	0, newArray, 0, 
							(detectors.length > newArray.length ? 
									newArray.length : detectors.length));
					detectors = newArray;
				}
			}else{
				detectors = new Detector[channels];
			}
	}
	
	final public void setChannelDetector(int channel, Detector detector) throws ArrayIndexOutOfBoundsException, NullPointerException{
		if (detector == null){
			throw new NullPointerException("Attempt to insert a null detector.");
		}
		synchronized (detectors){
			detectors[channel] = detector;
		}
	}

	/**
	 * Whoops.  this belongs in Fixture.
	 * 
	 * @param PImage
	 */
	final public void sync(PImage raster){
		synchronized (detectors){
			byte[] data = new byte[detectors.length];
			for (int i = 0; i < data.length; i++){
				if (detectors[i] == null){
					data[i] = 0;
				}else{
					// populate the pixel buffer
					// (if we need to optimize later, I can almost guarantee we
					//  should start here, by doing direct System.arrayCopy operations
					//  from raster.pixels.)
					PImage subraster = raster.get(detectors[i].x, 
													detectors[i].y, 
													detectors[i].width, 
													detectors[i].height);
					subraster.updatePixels();
					subraster.loadPixels();
					data[i] = (byte)detectors[i].model.getValue(subraster.pixels);
				}
			}
			send(data);
		}
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
	
	public Detector[] getDetectors(){
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
}