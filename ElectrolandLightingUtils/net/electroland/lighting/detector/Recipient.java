package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import processing.core.PImage;

/**
 * This is an abstract class that acts as a bridge between detectors, rasters,
 * and DMX lighting fixtures.  The only method that concrete implementations
 * need to implement is send(byte[]), which actually sends the bytes to the
 * fixture, using the appropriate protocol.  See ArtNetDMXLightingFixture.
 * @author geilfuss
 */
public abstract class Recipient 
{

	protected InetAddress ip;
	protected String ipStr, id;
	protected int port, channels;
	protected List <Detector> detectors;
	protected Dimension preferredDimensions; // for generating raster properly.
	protected String patchgroup;
	protected boolean detectorsOn = true;
	protected boolean log; 

	/**
	 * @param universe - the byte id of this lighting fixtures DMX universe
	 * @param ipStr - the ip address for the lighting controller for this fixture
	 * @param port - the port the lighting controller is listening on
	 * @param channels - the total number light channels this fixture is supporting
	 * @throws UnknownHostException
	 */
	public Recipient(String id, String ipStr, int port, int channels, Dimension preferredDimensions) throws UnknownHostException {
		this.ipStr = ipStr;
		this.port = port;

		this.id = id;
		this.setChannels(channels);
		this.preferredDimensions = preferredDimensions;

		this.ip = InetAddress.getByName(ipStr);
		this.detectors = Collections.synchronizedList(new ArrayList<Detector>(channels));
		this.channels = channels;
	}

	public Recipient(String id, String ipStr, int port, int channels, Dimension preferredDimensions, String patchgroup) throws UnknownHostException {
		this.ipStr = ipStr;
		this.port = port;

		this.id = id;
		this.setChannels(channels);
		this.preferredDimensions = preferredDimensions;

		this.ip = InetAddress.getByName(ipStr);
		this.detectors = Collections.synchronizedList(new ArrayList<Detector>(channels));
		this.channels = channels;
		this.patchgroup = patchgroup;
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

	final public void scale(double scaleDimensions){
		this.preferredDimensions.width = (int)(this.preferredDimensions.width * scaleDimensions);
		this.preferredDimensions.height = (int)(this.preferredDimensions.height * scaleDimensions);
	}
	
	/**
	 * 
	 * @param PImage
	 */
	final public void sync(PImage raster)
	{
		if(detectorsOn)
		{
			byte[] data = new byte[channels];

			for (int i = 0; i < data.length; i++)
			{
				data[i] = 0;
				
				if (i < detectors.size())
				{					
					Detector detector = detectors.get(i);

					if (detector != null)
					{
						PImage subraster = raster.get(detector.x, 
														detector.y, 
														detector.width, 
														detector.height);
						subraster.updatePixels();
						subraster.loadPixels();
						data[i] = (byte)detector.model.getValue(subraster.pixels);
					}
				}
			}
			send(data);
		}
	}

	/**
	 * @param image
	 */
	final public void sync(BufferedImage raster)
	{
		if(detectorsOn)
		{
			byte[] data = new byte[channels];

			for (int i = 0; i < data.length; i++)
			{
				data[i] = 0;// unnecessary?

				if (i < detectors.size())
				{
					Detector detector = detectors.get(i);
					
					if (detector != null)
					{
						int pixels[] = new int[detector.width * detector.height];

						raster.getRGB(detector.x, detector.y, 
									detector.width, detector.height, pixels, 
									0, detector.width);
						
						data[i] = (byte)detector.model.getValue(pixels);						
					}
				}
			}
			send(data);
		}
	}
	
	final public void toggleDetectors(){
		detectorsOn = !detectorsOn;
	}

	public int getChannels() {
		return channels;
	}
	
	public List<Detector> getDetectors(){
		return detectors;
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

	final public Dimension getPreferredDimensions()
	{
		return preferredDimensions;
	}

	public void setLog(boolean log){
		this.log = log;
	}
	
	public static String bytesToHex(byte[] b, int length){
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i< length; i++){
			sb.append(Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
		}
		return sb.toString();
	}
}