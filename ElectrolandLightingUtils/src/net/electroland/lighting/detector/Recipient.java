package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import net.electroland.lighting.detector.animation.Raster;
import net.electroland.util.Util;

import org.apache.log4j.Logger;

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
	static Logger logger = Logger.getLogger(Recipient.class);

	protected InetAddress ip;
	protected String ipStr, id;
	protected int port, channels;
	protected List <Detector> detectors;
	protected Map <Detector, Byte> lastEvals;
	protected Dimension preferredDimensions; // for generating raster properly.
	protected String patchgroup;
	protected ByteMap bytemap;
	protected boolean isEnabled = true;

	/**
	 * @param universe - the byte id of this lighting fixtures DMX universe
	 * @param ipStr - the ip address for the lighting controller for this fixture
	 * @param port - the port the lighting controller is listening on
	 * @param channels - the total number light channels this fixture is supporting
	 * @throws UnknownHostException
	 */
	public Recipient(String id, String ipStr, int port, int channels, 
					Dimension preferredDimensions) throws UnknownHostException
	{
		this.ipStr = ipStr;
		this.port = port;
		this.id = id;
		this.preferredDimensions = new Dimension(preferredDimensions);
		this.ip = InetAddress.getByName(ipStr);

		this.setChannels(channels);
	}

	public Recipient(String id, String ipStr, int port, int channels, 
					Dimension preferredDimensions, String patchgroup) throws UnknownHostException
	{
		this.ipStr = ipStr;
		this.port = port;
		this.id = id;
		this.preferredDimensions = new Dimension(preferredDimensions);
		this.ip = InetAddress.getByName(ipStr);
		this.setChannels(channels);

		this.patchgroup = patchgroup;
	}
	public void setByteMap(ByteMap bytemap)
	{
		this.bytemap = bytemap;
	}
	private void setChannels(int channels)
	{
		this.channels = channels;
		this.detectors = Collections.synchronizedList(new ArrayList<Detector>(channels));
		lastEvals = new HashMap<Detector, Byte>();
		for (int i = 0; i < channels; i++){
			detectors.add(null);
		}
	}

	/**
	 * Implement this.
	 * 
	 * @param data
	 */
	abstract void send(byte[] data);

	final public void setChannelDetector(int channel, Detector detector) throws ArrayIndexOutOfBoundsException, NullPointerException
	{
		detectors.set(channel, detector);
		lastEvals.put(detector, new Byte((byte)0));
	}

	final public void scale(double scaleDimensions)
	{
		this.preferredDimensions.width = (int)(this.preferredDimensions.width * scaleDimensions);
		this.preferredDimensions.height = (int)(this.preferredDimensions.height * scaleDimensions);
	}

	/**
	 * @param raster
	 */
	final public void sync(Raster raster)
	{
		this.sync(raster, null, null);
	}

	final public void blackOut()
	{
		send(new byte[channels]);

		// reflect the black state in the detectors (for GUIs relying on that
		// to render the latest state)
		Iterator <Detector> dIter = detectors.iterator();
		while (dIter.hasNext())
		{
			lastEvals.put(dIter.next(), new Byte((byte)0));
		}
	}

	/**
	 * 
	 * @param currentRaster
	 * @param transitionRaster
	 * @param targetRaster
	 */
	final public void sync(Raster currentRaster, Raster transitionRaster, Raster targetRaster)
	{
		boolean isSingle = transitionRaster == null && targetRaster == null;
		byte[] data = new byte[channels];

		for (int i = 0; i < data.length; i++)
		{
			if (i < detectors.size())
			{
				Detector detector = detectors.get(i);

				if (detector != null)
				{
					if (isSingle)
					{
						data[i]  = detect(currentRaster, detector);
					}else
					{
						data[i] = transition(detect(currentRaster, detector),
												detect(transitionRaster, detector),
												detect(targetRaster, detector));
					}
					lastEvals.put(detector, new Byte(data[i]));
				}
			}
		}
		if (bytemap != null)
		{
			data = bytemap.map(data);
		}
		if (isEnabled)
		{
			send(data);
		}
	}

	public static String renderMap(byte[] b)
	{
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < b.length; i++)
		{
			sb.append(Util.bytesToHex(b[i]));
		}
		return sb.toString();
	}

	
	/**
	 * 
	 * @param raster
	 * @param detector
	 * @return
	 */
	final private static byte detect(Raster raster, Detector detector)
	{
		if (raster == null)		
		{
			return (byte)0;
		}else
		{
			if (raster.isJava2d())
			{
				// Java2D
				int pixels[] = new int[detector.width * detector.height];
				try{
					((BufferedImage)raster.getRaster()).getRGB(
								detector.x, detector.y, 
								detector.width, detector.height, pixels, 
								0, detector.width);
				}catch(ArrayIndexOutOfBoundsException e){
					// Coordinate out of bounds! (just ignore)
				}
				return (byte)detector.model.getValue(pixels);

			}else
			{
				// PROCESSING
				PImage subraster = ((PImage)raster.getRaster()).get(detector.x, 
																	detector.y, 
																	detector.width, 
																	detector.height);
				subraster.updatePixels();
				subraster.loadPixels();
				
				return (byte)detector.model.getValue(subraster.pixels);
			}
		}
	}

	/**
	 * alpha: if alpha == 0, current image shows.  if alpha == 255, target
	 * image shows.  anything else is a gradient in between.
	 * 
	 * @param current
	 * @param targetAmount
	 * @param target
	 * @return
	 */
	final private static byte transition(byte current, byte targetAmount, byte target)
	{
		if (((int)targetAmount) == 0)
		{
			return current;
		}else
		{
			int currentInt = Util.unsignedByteToInt(current);
			double percentOfTargetShown = (Util.unsignedByteToInt(targetAmount) / 255.0);
			int targetInt = Util.unsignedByteToInt(target);

			int composite = (int)(((1.0 - percentOfTargetShown)*currentInt) 
										+ (percentOfTargetShown * targetInt));

			return composite > 255 ? (byte)255 : (byte)composite;
		}
	}



	final public Byte getLastEvaluatedValue(Detector detector)
	{
		return this.lastEvals.get(detector);
	}

	final public int getChannels()
	{
		return channels;
	}

	final public List<Detector> getDetectorPatchList()
	{
		return detectors;
	}

	final public InetAddress getIp()
	{
		return ip;
	}

	final public String getIpStr()
	{
		return ipStr;
	}

	final public int getPort()
	{
		return port;
	}
	
	final public String getID()
	{
		return id;
	}

	final public Dimension getPreferredDimensions()
	{
		return preferredDimensions;
	}

	public String toString()
	{
		StringBuffer sb = new StringBuffer(id);
		sb.append("=Recipient[InetAddress=").append(ip);
		sb.append(",port=").append(port);
		sb.append(",channels=").append(channels);
		sb.append(",preferredDimensions=").append(preferredDimensions);
		sb.append(",patchgroup=").append(patchgroup);
		sb.append(",bytemap=").append(bytemap);
		//sb.append(',').append(detectors).append(']');
		sb.append(']');
		return sb.toString();
	}
}