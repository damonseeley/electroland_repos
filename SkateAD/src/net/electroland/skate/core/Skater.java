package net.electroland.skate.core;

import java.awt.Graphics;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;

import javax.imageio.ImageIO;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import net.electroland.utils.Util;

import org.apache.log4j.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class Skater implements Cloneable {

	public String name;
	public String[] soundList;
	public int soundNodeID;
	public SoundNode soundNode;

	public String fileName;
	public int worldDim;
	public int startTick;
	public int endTick;
	public int frameRate;
	public int ticksPerFrame;
	public double lengthSeconds;
	public int lengthFrames;
	public long startTime;
	public String spriteFile;
	public BufferedImage spriteImg;
	public int spriteSize;
	public boolean globalSound = false;

	public double cmConversion = 2.54;  //incoming files should have inch units, by default do
	public double canvasScale;

	public HashMap[] frameData;
	
	private static Logger logger = Logger.getLogger(SkateMain.class);

	public Skater(String skaterName, String animXML, String dim, double cScale, String sprt, int sprtSize, String[] sounds, boolean isGlSnd) {

		if (skaterName != null){
			name = skaterName;
		} else {
			name = "generic";
		}

		if (dim != null){
			worldDim = Integer.parseInt(dim);
		} else {
			worldDim = 1000;
		}

		canvasScale = cScale;

		if (soundList != null){
			name = skaterName;
		} else {
			soundList = sounds;
		}

		spriteFile = sprt;
		
		//spriteImg = new ImageIcon(sprite).getImage();
		
		spriteSize = sprtSize;
		globalSound = isGlSnd;

		
		BufferedImage img;
		try {
			//InputStream imgIS = new Util().getClass().getClassLoader().getResourceAsStream(spriteFile);
			img = ImageIO.read(new File(spriteFile));
			//int w = img.getWidth(null);
			//int h = img.getHeight(null);
			spriteImg = new BufferedImage(spriteSize, spriteSize, BufferedImage.TYPE_INT_ARGB);
			Graphics g = spriteImg.getGraphics();
			//g.drawImage(img, 0, 0, null);
			g.drawImage(img, 0, 0, spriteSize, spriteSize, null); // so scaling works

			//g.drawRect(x, y, width, height)
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			logger.info("ERROR: cannot find sprite image");
		}
		
	
		
	
		boolean debugOutput = false;

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

	/**
	 * if a custom duration was specified.
	 * @param lengthSeconds
	 */
	public void setLengthOverride(double lengthSeconds)
	{
		this.lengthSeconds = lengthSeconds;
	}
	
	/* -------- ANIMATION --------
	 *  --------------------------- */

	public double percentComplete = 0.0;
	public int curFrame = 0; //array index so start at 0
	public long elapsed;
	public boolean animComplete;

	/* start animation playback by recording the start time for later comparison */
	public void startAnim() {
		startTime = System.currentTimeMillis();
		Point2D.Double curPos = new Point2D.Double(getMetricPosNow()[0],getMetricPosNow()[1]);
		soundNodeID = SkateMain.soundControllerP5.newSoundNode(soundList[0], curPos, 1.0f, globalSound); //right now just offer the first sound
		//logger.info(name + " " + soundNodeID);
	}

	/* update the play head based on the amount of time elapsed */
	public void animate() {
		// prevent anything from happening if animation is already complete
		if (!animComplete) {
			elapsed = System.currentTimeMillis() - startTime;
			percentComplete = (elapsed/1000.0) / lengthSeconds;
			//logger.info(percentComplete * 100 + "%");
			curFrame = (int)(lengthFrames * percentComplete);
			//logger.info(curFrame);

			if (curFrame < lengthFrames - 1){ // TODO: make sure we don't allow 1 frame shows!
				//update the sound
				Point2D.Double curPos = getMetric2DPosNow();
				SkateMain.soundControllerP5.updateSoundNodeByID(soundNodeID, curPos, 1.0f);

				animComplete = false;
			} else {
				SkateMain.soundControllerP5.dellocateByID(soundNodeID, 1.0f);
				percentComplete = 1.0;
				animComplete = true;
			}
		} else {
			//do nothing until garbage collection
		}
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
	
	public boolean isGlobal() {
		if (soundNode.globalSound){
			return true;
		} else {
			return false;
		}
	}

	/*
	 * NOTE these Pos methods ALL deal with animation files which contain 
	 * pos data expressed in inches within the maxDim dims of the 3D scene
	 * eg: incoming file x in inches is pos within a 2000cm grid
	 *     is later 
	 */


	/* Return the pos value in centimeters for the current frame */
	public Point2D.Double getMetric2DPosNow(){

		// get the percent of the way between frames that we are.
		elapsed = System.currentTimeMillis() - startTime;
		percentComplete = (elapsed/1000.0) / lengthSeconds;
		double appxFrame = lengthFrames * percentComplete;
		double percent = appxFrame - curFrame;

		Point2D.Double one = new Point2D.Double(); // last quantized position
		Point2D.Double two = new Point2D.Double(); // next quantized position
		Point2D.Double pos = new Point2D.Double(); // tween position

		if (curFrame == appxFrame){
			// special case: no interpolation required
			pos.x = (Double)frameData[curFrame].get("x");
			pos.y = (Double)frameData[curFrame].get("y");
		}else{
			one.x = (Double)frameData[curFrame].get("x");
			one.y = (Double)frameData[curFrame].get("y");
			two.x = (Double)frameData[curFrame + 1].get("x");
			two.y = (Double)frameData[curFrame + 1].get("y");
		}
		// interpolate
		pos.x = (percent * (two.x - one.x)) + one.x;
		pos.y = (percent * (two.y - one.y)) + one.y;
		
		// do conversions
		pos.x *= cmConversion;
		pos.y *= cmConversion;
		pos.y *= -1;
		
		return pos;
	}

	public Point2D.Double getCanvas2DPosNow(){
		Point2D.Double cpos = getMetric2DPosNow();
		cpos.x = cpos.x * canvasScale;
		cpos.y = cpos.y * canvasScale;
		return cpos;
	}

	// Position getters below this line have not been tested

	// WARNING: NO TWEENING HERE!
	/* Return the pos value in centimeters for the current frame */
	public double[] getMetricPosNow(){
		double[] pos = new double[3];
		pos[0] = (Double)frameData[curFrame].get("x") * cmConversion;
		pos[1] = (Double)frameData[curFrame].get("y") * cmConversion * -1;
		pos[2] = (Double)frameData[curFrame].get("z") * cmConversion;
		return pos;
	}

	// THIS METHOD DOES NOT TWEEN
	/* Return the pos value in centimeters for given frame 
	public double[] getMetricPosAtFrame(int index){
		double[] pos = new double[3];
		pos[0] = (Double)frameData[index].get("x") * cmConversion;
		pos[1] = (Double)frameData[index].get("y") * cmConversion * -1;
		pos[2] = (Double)frameData[index].get("z") * cmConversion;
		return pos;
	}*/

	// THIS METHOD DOES NOT TWEEN
	/* Return the pos value in centimeters for given time 
	public double[] getMetricPosAtTime(double time){
		double[] pos = new double[3];
		// quantize to individual frames by using int for frame value
		int frame = (int)((time/lengthSeconds) * lengthFrames);
		pos[0] = (Double)frameData[frame].get("x") * cmConversion;
		pos[1] = (Double)frameData[frame].get("y") * cmConversion * -1;
		pos[2] = (Double)frameData[frame].get("z") * cmConversion;
		return pos;
	}*/




	/* -------- PLUMBING --------
	 *  --------------------------- */

	/*
	public static void main(String argv[]) {
		@SuppressWarnings("unused")
		Skater sx = new Skater(null, "180f_pos.xaf", null, 1.0, null);
	} */

	public Object clone() throws CloneNotSupportedException {
		return super.clone();
	}


}



