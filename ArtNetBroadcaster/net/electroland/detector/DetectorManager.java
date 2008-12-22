package net.electroland.detector;

import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.Iterator;
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
		String fixStr = props.getProperty("fixture" + i);
		while (fixStr != null && fixStr.length() != 0){

			DMXLightingFixture fixture = parseFixture(fixStr, fps);
			System.out.println("got fixture " + fixture.id + " in lightgroup " + fixture.lightgroup);
			fixtures.put(fixture.id, fixture);
		
			fixStr = props.getProperty("fixture" + (++i));			
		}

		// load detectors
		i = 0;
		String detectStr = props.getProperty("detector" + i);
		while (detectStr != null && detectStr.length() != 0){
			Detector detector = parseDetector(detectStr);
			detectors.put("detector" + i++, detector);

			// HACKY CRAP FOR LAFM
			
			// add this detector to any fixture that belongs to this light group
			Iterator <DMXLightingFixture> itr = fixtures.values().iterator();
			while (itr.hasNext()){
				DMXLightingFixture fixture = itr.next();
				if (fixture.lightgroup.equals(detector.lightgroup)){
					fixture.setChannelDetector(detector.channel, detector);
				}
			}
		
			detectStr = props.getProperty("detector" + i);			
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

	public Detector[] getDetectors(){
		Detector[] d = new Detector[detectors.size()];
		detectors.values().toArray(d);
		return d;		
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
	
	private static DMXLightingFixture parseFixture(String str, int fps) throws UnknownHostException {
		
		// example: fixture1, 1, 75, 256, 256, ARTNET, 10.7.88.50, lightgroup0
		
		ArtNetDMXLightingFixture fixture = null;;
		StringTokenizer st = new StringTokenizer(str, ", \t");

		String id = st.nextToken();
//		System.out.println("id=" + id);
		byte universe = (byte)Integer.parseInt(st.nextToken());
//		System.out.println("universe=" + universe);
		int channels = Integer.parseInt(st.nextToken());
//		System.out.println("channels=" + channels);
		int width = Integer.parseInt(st.nextToken());
//		System.out.println("width=" + width);
		int height = Integer.parseInt(st.nextToken());
//		System.out.println("height=" + height);
		String protocol = st.nextToken();
//		System.out.println("protocol=" + protocol);
	
		if (protocol.equalsIgnoreCase("artnet")){
			String ip = st.nextToken();
			fixture = new ArtNetDMXLightingFixture(id, universe, ip, channels, width, height);
		}
//		System.out.println("ipaddress=" + ip);		
		String lightgroup = st.nextToken(); // need a string parse for this ("...")
//		System.out.println("lightgroup=" + lightgroup);		
		
		if (fixture != null && fixture instanceof ArtNetDMXLightingFixture){
			fixture.setLog(fps == 1);
			fixture.lightgroup = lightgroup;
			return fixture;
		}else{
			throw new RuntimeException("Unknown fixture protocol.");
		}
	}

	private static Detector parseDetector(String str){
		
		// example detector0 = 0,0,2,2

		StringTokenizer st = new StringTokenizer(str, ", \t");
		
		int x = Integer.parseInt(st.nextToken());
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