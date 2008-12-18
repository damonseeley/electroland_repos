package net.electroland.detector;

import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Properties;
import java.util.StringTokenizer;

import net.electroland.detector.models.BlueDetectionModel;
import net.electroland.detector.models.GreenDetectionModel;
import net.electroland.detector.models.RedDetectionModel;
import net.electroland.detector.models.ThresholdDetectionModel;

/**
 * loads detectors and fixtures from a lighting properties file.
 * 
 * @author geilfuss
 */
public class DetectorManager {

	private HashMap <String, DMXLightingFixture> fixtures;
	private HashMap <String, Detector> detectors;
	private int fps;

	public DetectorManager(Properties props) throws UnknownHostException{

		fixtures = new HashMap<String, DMXLightingFixture>();
		detectors = new HashMap<String, Detector>();

		// fps
		fps = Integer.parseInt(props.getProperty("fps"));
		
		// load fixtures
		int i = 0;
		String fixture = props.getProperty("fixture" + i);
		while (fixture != null && fixture.length() != 0){
//			System.out.println("loading " + fixture);
			String id = "fixture" + (i++);
			DMXLightingFixture fix = parseFixture(fixture, id);
			System.out.println("got fixture " + id + " in lightgroup " + fix.lightgroup);
			fixtures.put(id, fix);
		
			fixture = props.getProperty("fixture" + i);			
		}

		// load detectors
		i = 0;
		String detector = props.getProperty("detector" + i);
		while (detector != null && detector.length() != 0){
//			System.out.println("loading " + detector);
			Detector detect = parseDetector(detector);
			detectors.put("detector" + i++, detect);

			// HACKY CRAP FOR LAFM
			
			// add this detector to any fixture that belongs to this light group
			Iterator <DMXLightingFixture> itr = fixtures.values().iterator();
			while (itr.hasNext()){
				DMXLightingFixture fix = itr.next();
				if (fix.lightgroup.equals(detect.lightgroup)){
					fix.setChannelDetector(detect.channel, detect);
				}
			}
		
			detector = props.getProperty("detector" + i);			
		}

		
		// tie lights, fixtures, and detectors together.
		i = 0;
		String light = props.getProperty("light" + (i++));
		while (light != null && light.length() != 0){

			// example: light0 = 1, detector0
//			System.out.println("loading " + light);
			StringTokenizer st = new StringTokenizer(light);
			int channel = Integer.parseInt(st.nextToken(" \t,"));
//			System.out.println("channel=" + channel);
			DMXLightingFixture f = fixtures.get(st.nextToken(" \t,"));
//			System.out.println("fixture=" + f);
			Detector d = detectors.get(st.nextToken(" \t,"));
//			System.out.println("detector=" + d);
			
			f.setChannelDetector(channel, d);
			
			light = props.getProperty("light" + (i++));	
		}
	}

	public int getFps() {
		return fps;
	}

	public DMXLightingFixture getFixture(String id){
		return fixtures.get(id);
	}

	// returning the array instead of the hashmap, so the user can't monkey with the hashmap.
	public DMXLightingFixture[] getFixtures(){
		DMXLightingFixture[] f = new DMXLightingFixture[fixtures.size()];
		fixtures.values().toArray(f);
		return f;
	}
	
	public String[] getFixtureIds(){
		String[] k = new String[fixtures.size()];
		fixtures.keySet().toArray(k);
		return k;
	}
	
	private static DMXLightingFixture parseFixture(String str, String id) throws UnknownHostException {
		
		// example: fixture0 = 1, 127.0.0.1:8000, 75, ArtNet
		
		StringTokenizer st = new StringTokenizer(str);

		byte universe = (byte)Integer.parseInt(st.nextToken(", \t"));
//		System.out.println("universe=" + universe);
		String ip = st.nextToken(", \t:");
//		System.out.println("ipaddress=" + ip);
		int port = Integer.parseInt(st.nextToken(", \t:"));
//		System.out.println("port=" + port);
		int channels = Integer.parseInt(st.nextToken(", \t"));
//		System.out.println("channels=" + channels);
		String protocol = st.nextToken();
//		System.out.println("protocol=" + protocol);
		int width = Integer.parseInt(st.nextToken(", \t"));
//		System.out.println("width=" + width);
		int height = Integer.parseInt(st.nextToken(", \t"));
//		System.out.println("height=" + height);
		
		String lightgroup = st.nextToken(", \t");
		
		if (protocol.equalsIgnoreCase("artnet")){
			ArtNetDMXLightingFixture fixture = new ArtNetDMXLightingFixture(universe, ip, port, channels, width, height, id);
			fixture.lightgroup = lightgroup;
			return fixture;
		}else{
			throw new RuntimeException("Unknown fixture protocol.");
		}
	}

	private static Detector parseDetector(String str){
		
		// example detector0 = 0,0,2,2

		StringTokenizer st = new StringTokenizer(str);
		
		int x = Integer.parseInt(st.nextToken(", \t"));
//		System.out.println("x=" + x);
		int y = Integer.parseInt(st.nextToken());
//		System.out.println("y=" + y);
		int w = Integer.parseInt(st.nextToken());
//		System.out.println("w=" + w);
		int h = Integer.parseInt(st.nextToken());
//		System.out.println("h=" + h);
		String dmName = st.nextToken();
		
		String lightgroup = st.nextToken();
		int channel = Integer.parseInt(st.nextToken());

//		Class detectionModel = DetectorManager.class.getClassLoader().loadClass(dmName);
//		DetectionModel = detectionModel.getMethods()....(need to find constructor)

		DetectionModel model = null;
		// need to implement a real class loader here.
		if (dmName.equalsIgnoreCase("RedDetectionModel")){
			model = new RedDetectionModel();
		}else if (dmName.equalsIgnoreCase("GreenDetectionModel")){
			model = new GreenDetectionModel();
		}else if (dmName.equalsIgnoreCase("BlueDetectionModel")){
			model = new BlueDetectionModel();
		}else if (dmName.equalsIgnoreCase("ThresholdDetectionModel")){
			model = new ThresholdDetectionModel();
		}

//		System.out.println(model);
		Detector d = new Detector(x,y,w,h,model);
		d.lightgroup = lightgroup;
		d.channel = channel;

		return d;
	}
}