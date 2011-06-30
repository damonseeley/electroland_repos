package net.electroland.skate.core;

import java.io.File;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.HashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.Util;

import org.apache.log4j.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class Skater implements Cloneable {

	public String name;
	public String[] soundList;
	public int amplitude;
	public int soundNode;

	public String fileName;
	public int maxDim;
	public int startTick;
	public int endTick;
	public int frameRate;
	public int ticksPerFrame;
	public double lengthSeconds;
	public int lengthFrames;
	public long startTime;

	public double cmConversion;

	public HashMap[] frameData;
	
	private static Logger logger = Logger.getLogger(SkateMain.class);

	public Skater(String skaterName, String animXML, String dim, String[] sounds) {
		
		if (skaterName != null){
			name = skaterName;
		} else {
			name = "generic";
		}
		
		if (dim != null){
			maxDim = Integer.parseInt(dim);
		} else {
			maxDim = 1000;
		}
		
		if (soundList != null){
			name = skaterName;
		} else {
			soundList = sounds;
		}
		
		cmConversion = 2.54; //incoming files should have inch units, by default do
		boolean debugOutput = true;

		try {

			//File fXmlFile = new File(animXML);
			InputStream fXmlFile = new Util().getClass().getClassLoader().getResourceAsStream(animXML);
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(fXmlFile);
			doc.getDocumentElement().normalize();

			// SETUP SCENE INFO
			if (debugOutput){
				logger.info("---------- SCENE INFO ----------");
			}

			// Parse out SceneInfo attribs into object vars
			NodeList SIlist = doc.getElementsByTagName("SceneInfo");
			//logger.info(SIlist.getLength());
			//logger.info(SIlist.item(0).getNodeName());

			Element SIel = (Element)SIlist.item(0);
			fileName = SIel.getAttribute("fileName");
			startTick = Integer.parseInt(SIel.getAttribute("startTick"));
			endTick = Integer.parseInt(SIel.getAttribute("endTick"));
			frameRate = Integer.parseInt(SIel.getAttribute("frameRate"));
			ticksPerFrame = Integer.parseInt(SIel.getAttribute("ticksPerFrame"));

			//use number of samples(frames) to determine length in seconds
			NodeList nl = doc.getElementsByTagName("S");		
			lengthFrames = nl.getLength();
			lengthSeconds = (double)nl.getLength()/(double)frameRate;

			if (debugOutput){
				logger.info("origin fileName: " + fileName);
				logger.info("startTick: " + startTick);
				logger.info("endTick: " + endTick);
				logger.info("frameRate: " + frameRate);
				logger.info("ticksPerFrame: " + ticksPerFrame);
				logger.info("length (in frames): " + lengthFrames);
				logger.info("length (in seconds): " + lengthSeconds);
				logger.info("-------- END SCENE INFO --------");
				logger.info("---------- ANIM DATA ----------");
			}

			//create an array of hashtables to store frame data
			frameData  = new HashMap[nl.getLength()];

			//get the sample data from XML - note we already created the NodeList above for "S"
			if(nl != null && nl.getLength() > 0) {
				for(int i = 0 ; i < nl.getLength();i++) {

					//create a new frame hashtable
					HashMap<String,Double> ht = new HashMap<String,Double>();

					//get the sample element
					Element el = (Element)nl.item(i);

					//store t as the first hash pair
					ht.put("ticks", Double.parseDouble(el.getAttribute("t")));

					//get the v string and do stringops on it
					String v = el.getAttribute("v");
					String[] vs = v.split(" ");

					//put the v values into our hashmap
					ht.put("z", Double.parseDouble(vs[11]));
					ht.put("y", Double.parseDouble(vs[10]));
					ht.put("x", Double.parseDouble(vs[9]));

					frameData[i] = ht;
					//logger.info(i);
				}
			}

			
			//test print
			//logger.info(frameData[96]);
			if (debugOutput){
				logger.info("-------- END ANIM DATA --------");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	
	/* -------- ANIMATION --------
	 *  --------------------------- */
	
	
	/* start animation playback by recording the start time for later comparison */
	public void startAnim() {
		startTime = System.currentTimeMillis();
		int startX = (int) getMetricPosNow()[0];
		int startY = (int) getMetricPosNow()[1];
		soundNode = SkateMain.soundController.newSoundNode(soundList[0], startX, startY, 1.0f, soundList[0]);
		logger.info(name + " " + soundNode);
	}
	
	public double percentComplete = 0.0;
	public int curFrame = 0; //array index so start at 0
	public long elapsed;
	public boolean animComplete;
	
	/* update the play head based on the amount of time elapsed */
	public void animate() {
		elapsed = System.currentTimeMillis() - startTime;
		percentComplete = (elapsed/1000.0) / lengthSeconds;
		//logger.info(percentComplete * 100 + "%");
		curFrame = (int)(lengthFrames * percentComplete);
		//logger.info(curFrame);
		
		if (curFrame < lengthFrames){
			animComplete = false;
		} else {
			animComplete = true;
		}
		//logger.info(amplitude);
		
	}
	
	

	
	
	
	/* -------- GETTERS --------
	 *  --------------------------- */
	
	public boolean isLive() {
		if (animComplete){
			return true;
		} else {
			return false;
		}
	}

	/* Return the pos value in centimeters for the current frame */
	public double[] getMetricPosNow(){
		double[] pos = new double[3];
		pos[0] = (Double)frameData[curFrame].get("x") * cmConversion;
		pos[1] = (Double)frameData[curFrame].get("y") * cmConversion;
		pos[2] = (Double)frameData[curFrame].get("z") * cmConversion;
		return pos;
	}
	
	/* Return the pos value in centimeters for given frame */
	public double[] getMetricPosAtFrame(int index){
		double[] pos = new double[3];
		pos[0] = (Double)frameData[index].get("x") * cmConversion;
		pos[1] = (Double)frameData[index].get("y") * cmConversion;
		pos[2] = (Double)frameData[index].get("z") * cmConversion;
		return pos;
	}


	/* Return the pos value in centimeters for given time */
	public double[] getMetricPosAtTime(double time){
		double[] pos = new double[3];
		// quantize to individual frames by using int for frame value
		int frame = (int)((time/lengthSeconds) * lengthFrames);
		pos[0] = (Double)frameData[frame].get("x") * cmConversion;
		pos[1] = (Double)frameData[frame].get("y") * cmConversion;
		pos[2] = (Double)frameData[frame].get("z") * cmConversion;
		return pos;
	}
	
	
	
	
	/* -------- PLUMBING --------
	 *  --------------------------- */


	public static void main(String argv[]) {
		@SuppressWarnings("unused")
		Skater sx = new Skater(null, "180f_pos.xaf", null, null);
	}

	public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }


}



