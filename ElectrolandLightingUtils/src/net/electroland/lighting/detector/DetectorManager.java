package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.Vector;

import net.electroland.lighting.detector.models.BlueDetectionModel;
import net.electroland.lighting.detector.models.BrightRedDetectionModel;
import net.electroland.lighting.detector.models.GreenDetectionModel;
import net.electroland.lighting.detector.models.RedDetectionModel;
import net.electroland.lighting.detector.models.ThresholdDetectionModel;
import net.electroland.util.OptionException;
import net.electroland.util.OptionParser;

import org.apache.log4j.Logger;

/**
 * loads detectors and fixtures from a lighting properties file.
 * 
 * @author geilfuss
 */
public class DetectorManager {

	static Logger logger = Logger.getLogger(DetectorManager.class);

	final static boolean ON = true;
	final static boolean OFF = false;
	
	private Map <String, Dimension> rasters;
	private Map <String, Recipient> recipients;
	private Map <String, Detector> detectors;
	private Map <String, ByteMap> bytemaps;
	private int fps;
	private File propsFile;

	public File getPropsFile() {
		return propsFile;
	}

	public void setPropsFile(File propsFile) {
		this.propsFile = propsFile;
	}

	public DetectorManager(String propsFileName) throws UnknownHostException, IOException,  UnknownHostException, OptionException
	{
		propsFile = new File(propsFileName);
		init(propsFile);
	}

	public DetectorManager(Properties props) throws UnknownHostException, OptionException
	{
		init(props);
	}

	public void saveChanges(boolean backup)
	{
		// create a new props file based on the objects in memory.
	}
	
	public void init(File file) throws IOException, OptionException
	{
		Properties systemProps = new Properties();
		systemProps.load(new FileInputStream(file));
		init(systemProps);		
	}
	
	public void init(Properties props) throws UnknownHostException, OptionException
	{
		bytemaps = Collections.synchronizedMap(new HashMap<String, ByteMap>());
		rasters = Collections.synchronizedMap(new HashMap<String, Dimension>());
		recipients = Collections.synchronizedMap(new HashMap<String, Recipient>());
		detectors = Collections.synchronizedMap(new HashMap<String, Detector>());

		// fps
		fps = Integer.parseInt(props.getProperty("fps"));

		double scalePositions = 1.0;

		// scale factor for x,y of detectors
		if (props.getProperty("detectorPositionScaling") != null && 
			props.getProperty("detectorPositionScaling").length() > 0)
		{
			scalePositions = Double.parseDouble(props.getProperty("detectorPositionScaling"));
		}
		logger.info("detectorPositionScaling=" + scalePositions);

		double scaleDimensions = 1.0;

		// scale factor for width,height of detectors
		if (props.getProperty("detectorDimensionScaling") != null && 
			props.getProperty("detectorDimensionScaling").length() > 0)
		{
			scaleDimensions = Double.parseDouble(props.getProperty("detectorDimensionScaling"));
		}
		logger.info("detectorDimensionScaling=" + scaleDimensions);

		double scaleRaster = 1.0;

		// scale factor for rasters
		if (props.getProperty("rasterDimensionScaling") != null && 
			props.getProperty("rasterDimensionScaling").length() > 0)
		{
			scaleRaster = Double.parseDouble(props.getProperty("rasterDimensionScaling"));
		}
		logger.info("rasterDimensionScaling=" + scaleRaster);

		// load rasters
		Enumeration <Object> e = props.keys();
		while (e.hasMoreElements())
		{
			String key = ("" + e.nextElement()).trim();
			if (key.toLowerCase().startsWith("raster."))
			{
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					String id = key.substring(idStart + 1, key.length());
					Dimension raster = parseRaster(id, "" + props.get(key));
					logger.info("raster." + id + "=" + raster);
					rasters.put(id, raster);
				}
			}
		}

		// load bytemaps
		Enumeration <Object> b = props.keys();
		while (b.hasMoreElements())
		{
			String key = ("" + b.nextElement()).trim();
			if (key.toLowerCase().startsWith("bytemap."))
			{
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					String id = key.substring(idStart + 1, key.length());
					ByteMap map = parseByteMap(id, "" + props.get(key));
					logger.info("bytemap." + id + "=" + map);
					bytemaps.put(id, map);
				}
			}
		}


		// load recipient
		Enumeration <Object> f = props.keys();
		while (f.hasMoreElements())
		{
			String key = ("" + f.nextElement()).trim();
			if (key.toLowerCase().startsWith("recipient."))
			{
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					String id = key.substring(idStart + 1, key.length());
					Recipient recipient = parseRecipient(id, "" + props.get(key), this);
					recipient.scale(scaleRaster);
					recipients.put(id, recipient);
				}
			}
		}

		// load detectors
		Enumeration <Object> g = props.keys();
		while (g.hasMoreElements())
		{
			String key = ("" + g.nextElement()).trim();
			if (key.toLowerCase().startsWith("detector."))
			{
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					String id = key.substring(idStart + 1, key.length());
					Detector detector = parseDetector(id, "" + props.get(key));
					detector.scale(scalePositions, scaleDimensions);
					logger.info("detector." + id + "=" + detector);
					detectors.put(id, detector);

					// add this detector to any fixture that belongs to this light group
					if (detector.patchgroup != null){
						Iterator <Recipient> itr = recipients.values().iterator();
						while (itr.hasNext()){
							Recipient recipient = itr.next();
							if (recipient.patchgroup != null && 
								recipient.patchgroup.equals(detector.patchgroup)){
								recipient.setChannelDetector(detector.channel, detector);
							}
						}
					}
				}
			}
		}

		// load patches
		Enumeration <Object> h = props.keys();
		while (h.hasMoreElements())
		{
			String key = ("" + h.nextElement()).trim();
			if (key.toLowerCase().startsWith("patch."))
			{
				int idStart = key.indexOf('.');
				if (idStart == -1 || idStart == key.length() - 1)
				{
					throw new OptionException("no id specified in property " + key);
				}else{
					String id = key.substring(idStart + 1, key.length());
					Patch patch = parsePatch(id, "" + props.get(key));
					Recipient recipient = recipients.get(patch.recipient);
					Detector detector = detectors.get(patch.detector);
					// TO DO: throw an error if recipient or detector are null.
					recipient.setChannelDetector(patch.channel, detector);
				}
			}
		}

		Iterator<Recipient> itr = recipients.values().iterator();
		while (itr.hasNext()){
			logger.info("recipient." + itr.next());
		}

		Runtime.getRuntime().addShutdownHook(new BlackOutThread(this));
	}

	// maps any given byte map to a different value.
	final private static ByteMap parseByteMap(String id, String str) throws OptionException
	{
		byte[] bytes = new byte[256];

		Map<String,Object> options = OptionParser.parse(str);
		// parse 256 values
		String strFullMap = getOption(options, "-fullmap", id, true);
		StringTokenizer st = new StringTokenizer(strFullMap, ",[] \t"); // bullshit parsing.
		int ptr = 0;
		while (st.hasMoreTokens())
		{
			if (ptr == 256)
			{
				throw new OptionException(id + ": -fullmap contains more than 256 values");
			}
			bytes[ptr++] = (byte)(Integer.parseInt(st.nextToken()));
		}
		return new ByteMap(bytes);
	}

	final private static Dimension parseRaster(String id, String str) throws OptionException
	{
		Map<String,Object> options = OptionParser.parse(str);
		//	raster.id = -width int -height int
		int width = Integer.parseInt(getOption(options, "-width", id, true));
		int height = Integer.parseInt(getOption(options, "-height", id, true));
		return new Dimension(width, height);
	}

	final public static Recipient parseRecipient(String id, String str, DetectorManager dmr) throws OptionException, UnknownHostException
	{
		Recipient r;
		Map<String,Object> options = OptionParser.parse(str);
		String protocol = getOption(options, "-protocol", id, true);
		int channels = Integer.parseInt(getOption(options, "-channels", id, true));
		Dimension d = dmr.rasters.get(getOption(options, "-defaultRaster", id, true));
		String patchgroup = getOption(options, "-patchgroup", id, false);
		ByteMap map = dmr.bytemaps.get(getOption(options, "-bytemap", id, false));

		if ("ARTNET_DOUBLE".equalsIgnoreCase(protocol))
		{
			//	recipient.id = -protocol ARTNET_DOUBLE -channels int -address string -universe int -defaultRaster raster.id [-patchgroup string]
			String address = getOption(options, "-address", id, true);
			byte universe = (byte)Integer.parseInt(getOption(options, "-universe", id, true));

			r = new ArtNetDoubleByteRecipient(id, universe, address, channels, d, patchgroup);

		}else if ("ARTNET".equalsIgnoreCase(protocol))
		{
			//	recipient.id = -protocol ARTNET -channels int -address string -universe int -defaultRaster raster.id [-patchgroup string]
			String address = getOption(options, "-address", id, true);
			byte universe = (byte)Integer.parseInt(getOption(options, "-universe", id, true));

			r = new ArtNetRecipient(id, universe, address, channels, d, patchgroup);

		}else if ("HaleUDP".equalsIgnoreCase(protocol)){
			//	recipient.id = -protocol HALEUDP -channels int -address string -port int -defaultRaster raster.id [-patchgroup string]
			String address = getOption(options, "-address", id, true);
			int port = Integer.parseInt(getOption(options, "-port", id, true));
			String interCmdStr = getOption(options, "-interCmdByte", id, false);
			Byte intrCmdByte = interCmdStr == null ? null : new Byte((byte)(Integer.parseInt(interCmdStr, 16)));
			Integer interPeriod = Integer.parseInt(getOption(options, "-interPeriod", id, false));

			r = new HaleUDPRecipient(id, address, port, channels, d, patchgroup, intrCmdByte, interPeriod);

		} else if ("FlexXML".equalsIgnoreCase(protocol)){
			//	recipient.id = -protocol FLEXXML -channels int -port int -defaultRaster raster.id [-patchgroup string]
			int port = Integer.parseInt(getOption(options, "-port", id, true));

			r = new FlexXMLRecipient(id, port, channels, d, patchgroup);

		}else {
			throw new OptionException("no such protocol " + protocol + " in recipient " + id);
		}

		r.originalStr = str;
		r.setByteMap(map);
		return r;
	}

	final private static Detector parseDetector(String id, String str) throws OptionException
	{
		Map<String,Object> options = OptionParser.parse(str);
		//	detector.id = -boundary boundaryObject -model modelClasspath [-patchgroup string int]
		Rectangle boundary = (Rectangle) parseBoundary(getOption(options, "-boundary", id, true));
		String modelStr = getOption(options, "-model", id, true);
		DetectionModel model = null;
/*
		try
		{
		    model = (DetectionModel)(Class.forName(modelStr).newInstance());
		}catch(ClassNotFoundException e)
		{
		    throw new OptionException("Can't find DetectionModel " + modelStr + " in " + id);
		}catch(ClassCastException e)
		{
		    throw new OptionException(modelStr + " is not a valid DetectionModel in " + id);
		} catch (InstantiationException e) 
		{
			throw new OptionException(modelStr + " InstantiationException in " + id);
		} catch (IllegalAccessException e) 
		{
			throw new OptionException(modelStr + " IllegalAccessException in " + id);
		}		
*/				
		// need to implement a real class loader here.
		if (modelStr.equalsIgnoreCase("net.electroland.lighting.detector.models.RedDetectionModel")){
			model = new RedDetectionModel();
		}else if (modelStr.equalsIgnoreCase("net.electroland.lighting.detector.models.GreenDetectionModel")){
			model = new GreenDetectionModel();
		}else if (modelStr.equalsIgnoreCase("net.electroland.lighting.detector.models.BlueDetectionModel")){
			model = new BlueDetectionModel();
		}else if (modelStr.equalsIgnoreCase("net.electroland.lighting.detector.models.ThresholdDetectionModel")){
			model = new ThresholdDetectionModel();
		}else if (modelStr.equalsIgnoreCase("net.electroland.lighting.detector.models.BrightRedDetectionModel")){
			model = new BrightRedDetectionModel();
		}

		String patchgroupStr = getOption(options, "-patchgroup", id, false);
		
		if (patchgroupStr != null)
		{
			int s = patchgroupStr.indexOf(' ');
			if (s == -1)
			{
				throw new OptionException("detector patchgroup for " + id + " requires both patchgroupg and channel");
			}else{
				String patchgroup = patchgroupStr.substring(0, s);
				int channel = Integer.parseInt(patchgroupStr.substring(s + 1, patchgroupStr.length()));
				return new Detector(boundary.x, boundary.y, boundary.width, boundary.height, 
									model, patchgroup, channel);
			}
		}else
		{
			return new Detector(boundary.x, boundary.y, boundary.width, boundary.height, 
					model);
			
		}
	}

	final private static Object parseBoundary(String str) throws OptionException
	{
		if (str.toLowerCase().startsWith("rectangle(")){
			str = str.substring(10, str.length());
			StringTokenizer st = new StringTokenizer(str, " \t,)");
			try{
				return new Rectangle(Integer.parseInt(st.nextToken()),
										Integer.parseInt(st.nextToken()),
										Integer.parseInt(st.nextToken()),
										Integer.parseInt(st.nextToken()));
			}catch(NumberFormatException e){
				throw new OptionException("incorrect arguments for boundary: " + str);
			}
		}else{
			throw new OptionException("unknown boundary type " + str);
		}
	}

	final private static Patch parsePatch(String id, String str) throws OptionException
	{
		Map<String,Object> options = OptionParser.parse(str);
		int channel = Integer.parseInt(getOption(options, "-channel", id, true));
		String detector = getOption(options, "-detector", id, true);
		String recipient = getOption(options, "-recipient", id, true);
		return new Patch(channel, detector, recipient);
	}

	final private static String getOption(Map<String, Object> options, String key, String id, boolean isMandatory) throws OptionException
	{
		Object o = options.get(key);
		if (o == null && isMandatory)
		{
			throw new OptionException("no " + key + " for recipient " + id);
		}else{
			return o == null ? null : o.toString();
		}
	}

	public int getFps() 
	{
		return fps;
	}

	public Recipient getRecipient(String id)
	{
		return recipients.get(id);
	}

	// returns a point in time copy of the detectors
	public Collection<Detector> getDetectors()
	{
		return new Vector<Detector>(detectors.values());
	}

	// returns a point in time copy of the recipients
	public Collection<Recipient> getRecipients()
	{
		return new Vector<Recipient>(recipients.values());
	}

	public String[] getRecipientIds()
	{
		String[] k = new String[recipients.size()];
		recipients.keySet().toArray(k);
		return k;
	}

	/**
	 * @deprecated
	 */
	public void blackOutAll()
	{
		Iterator<Recipient> i = recipients.values().iterator();
		while (i.hasNext())
		{
			Recipient r = i.next();
			logger.info("black out " + r.id);
			r.blackOut();
		}
	}

	public void allOff()
	{
		Iterator<Recipient> i = recipients.values().iterator();
		while (i.hasNext())
		{
			Recipient r = i.next();
			logger.info("off " + r.id);
			r.allOff();
		}
	}
	public void allOn()
	{
		Iterator<Recipient> i = recipients.values().iterator();
		while (i.hasNext())
		{
			Recipient r = i.next();
			logger.info("on " + r.id);
			r.allOn();
		}		
	}
	
	public void turnOff()
	{
		this.setDisplayState(OFF);
		blackOutAll();
	}


	public void turnOn()
	{
		this.setDisplayState(ON);
	}

	private void setDisplayState(boolean isEnabled)
	{
		Iterator<Recipient> rIter = recipients.values().iterator();
		while (rIter.hasNext())
		{
			rIter.next().isEnabled = isEnabled;
		}
	}
	
}
class BlackOutThread extends Thread{
	private DetectorManager dmr;
	public BlackOutThread(DetectorManager dmr)
	{
		this.dmr = dmr;
	}
	public void run()
	{
		dmr.turnOff();
	}
}
class Patch{
	public Patch(int channel, String detector, String recipient)
	{
		this.channel = channel;
		this.detector = detector;
		this.recipient = recipient;
	}
	int channel;
	String recipient, detector;
}