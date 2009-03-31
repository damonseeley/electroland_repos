package net.electroland.detector;

import java.net.UnknownHostException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
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

	private Map <String, DMXLightingFixture> fixtures;
	private Map <String, Detector> detectors;
	private int fps;
	
	public DetectorManager(Properties props) throws UnknownHostException{

		fixtures = Collections.synchronizedMap(new HashMap<String, DMXLightingFixture>());
		detectors = Collections.synchronizedMap(new HashMap<String, Detector>());

		// fps
		fps = Integer.parseInt(props.getProperty("fps"));
		
		double scalePositions = 1.0;
		
		// scale factor for x,y of detectors
		if (props.getProperty("scalePositions") != null && props.getProperty("scalePositions").length() > 0){
			scalePositions = Double.parseDouble(props.getProperty("scalePositions"));
		}

		double scaleDimensions = 1.0;
		
		// scale factor for x,y of detectors
		if (props.getProperty("scaleDimensions") != null && props.getProperty("scaleDimensions").length() > 0){
			scaleDimensions = Double.parseDouble(props.getProperty("scaleDimensions"));
		}
		
//		System.out.println("scalePositions=" + scalePositions);
//		System.out.println("scaleDimensions=" + scaleDimensions);

		// load fixtures
		int i = 0;
		String fixStr = props.getProperty("fixture" + i);
		while (fixStr != null && fixStr.length() != 0){

			DMXLightingFixture fixture = parseFixture(fixStr, fps);
			// somewhat non-intuitive:  the fixture scales by the position of the detectors.
			// (scale dimensions scales the detector dimensions)
			fixture.scale(scalePositions);
			//System.out.println("got fixture " + fixture.id + " in lightgroup " + fixture.lightgroup);
			fixtures.put(fixture.id, fixture);
		
			fixStr = props.getProperty("fixture" + (++i));			
		}

		// load detectors
		i = 0;
		String detectStr = props.getProperty("detector" + i);
		while (detectStr != null && detectStr.length() != 0){
			Detector detector = parseDetector(detectStr);
			detector.scale(scalePositions, scaleDimensions);
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
			StringTokenizer st = new StringTokenizer(light);
			int channel = Integer.parseInt(st.nextToken(" \t,"));
			DMXLightingFixture f = fixtures.get(st.nextToken(" \t,"));
			Detector d = detectors.get(st.nextToken(" \t,"));
			
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

	public Collection<Detector> getDetectors(){
		return detectors.values();
	}
	
	// returning the array instead of the hashmap, so the user can't monkey with the hashmap.
	public Collection<DMXLightingFixture> getFixtures(){
		return fixtures.values();
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
		byte universe = (byte)Integer.parseInt(st.nextToken());
		int channels = Integer.parseInt(st.nextToken());
		int width = Integer.parseInt(st.nextToken());
		int height = Integer.parseInt(st.nextToken());
		String protocol = st.nextToken();
	
		if (protocol.equalsIgnoreCase("artnet")){
			String ip = st.nextToken();
			fixture = new ArtNetDMXLightingFixture(id, universe, ip, channels, width, height);
		}
		String lightgroup = st.nextToken(); // need a string parse for this ("...")
		//String color = st.nextToken();
		//int soundChannel = Integer.parseInt(st.nextToken());
		
		if (fixture != null && fixture instanceof ArtNetDMXLightingFixture){
			fixture.setLog(fps == 1);
			fixture.lightgroup = lightgroup;
			//fixture.color = color;				// THIS DATA HAS BEEN MOVED TO physicalProps
			//fixture.soundChannel = soundChannel;
			return fixture;
		}else{
			throw new RuntimeException("Unknown fixture protocol.");
		}
	}

	private static Detector parseDetector(String str){
		
		// example detector0 = 0,0,2,2

		StringTokenizer st = new StringTokenizer(str, ", \t");
		
		int x = Integer.parseInt(st.nextToken());
		int y = Integer.parseInt(st.nextToken());
		int w = Integer.parseInt(st.nextToken());
		int h = Integer.parseInt(st.nextToken());
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

		Detector d = new Detector(x,y,w,h,model);
		d.lightgroup = lightgroup;
		d.channel = channel;

		return d;
	}
}