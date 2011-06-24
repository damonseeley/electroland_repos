package net.electroland.skate.core;

import java.io.File;
import java.lang.reflect.Array;
import java.util.HashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class Skater {

	public String fileName;
	public int startTick;
	public int endTick;
	public int frameRate;
	public int ticksPerFrame;
	public double lengthSeconds;
	public int lengthFrames;

	public double cmConversion;

	public HashMap[] frameData;


	public Skater(String XMLfile) {

		cmConversion = 2.54; //incoming files should have inch units, by default do
		boolean debugOutput = false;

		try {

			File fXmlFile = new File(XMLfile);
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(fXmlFile);
			doc.getDocumentElement().normalize();

			// SETUP SCENE INFO
			if (debugOutput){
			System.out.println("---------- SCENE INFO ----------");
			}

			// Parse out SceneInfo attribs into object vars
			NodeList SIlist = doc.getElementsByTagName("SceneInfo");
			//System.out.println(SIlist.getLength());
			//System.out.println(SIlist.item(0).getNodeName());

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
				System.out.println("origin fileName: " + fileName);
				System.out.println("startTick: " + startTick);
				System.out.println("endTick: " + endTick);
				System.out.println("frameRate: " + frameRate);
				System.out.println("ticksPerFrame: " + ticksPerFrame);
				System.out.println("length (in frames): " + lengthFrames);
				System.out.println("length (in seconds): " + lengthSeconds);
				System.out.println("-------- END SCENE INFO --------");
				System.out.println("---------- ANIM DATA ----------");
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
					//System.out.println(i);
				}
			}

			
			//test print
			//System.out.println(frameData[96]);
			if (debugOutput){
				System.out.println("-------- END ANIM DATA --------");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		//System.out.println(getMetricPosAtTime(3.2)[0] + " " + getMetricPosAtTime(3.2)[1] + " " + getMetricPosAtTime(3.2)[2] + " ");

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


	public static void main(String argv[]) {
		@SuppressWarnings("unused")
		Skater sx = new Skater("depends//180f_pos.xaf");
	}


}



