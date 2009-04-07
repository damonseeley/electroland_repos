package net.electroland.laface.shows;

import java.io.File;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.core.SpriteListener;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class WaveShow implements Animation, SpriteListener{
	
	private Raster r;
	private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	private ConcurrentHashMap<Integer,Wave> waves;			// used to manage properties of waves from control panel
	private int spriteIndex = 0;
	private int waveCount = 3;
	private int brightness = 255;		// used for tinting waves
	private boolean mirror = false;	// set to true when simply mirroring activity in another show

	public WaveShow(Raster r){
		this.r = r;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		waves = new ConcurrentHashMap<Integer,Wave>();
		// TODO getting shared wave sprite for now
		/*
		try{
			loadSprites();	// attempt to load waves from file
		} catch (Exception e) {
			for(int i=0; i<waveCount; i++){
				Wave wave = new Wave(spriteIndex, r, 0, 0);
				wave.addListener(this);
				wave.setAlpha(100);
				sprites.put(spriteIndex, wave);
				waves.put(spriteIndex, wave);
				spriteIndex++;
			}
		}
		*/
	}
	
	private void loadSprites() throws Exception{
		File spriteFile = new File("depends//waves.properties");
		DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
		DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
		Document doc = dBuilder.parse(spriteFile);
		doc.getDocumentElement().normalize();
		NodeList nList = doc.getElementsByTagName("wave");
		for(int i=0; i<nList.getLength(); i++){
			Node nNode = nList.item(i);
			if (nNode.getNodeType() == Node.ELEMENT_NODE) {
				Element element = (Element) nNode;
				Wave wave = new Wave(spriteIndex, r, 0, 0);
				wave.addListener(this);
				wave.setDamping(Double.parseDouble(getTagValue("damping",element)));
				wave.setNonlinearity(Double.parseDouble(getTagValue("nonlinearity",element)));
				wave.setYoffset(Double.parseDouble(getTagValue("yoffset",element)));
				wave.setDX(Double.parseDouble(getTagValue("dx",element)));
				wave.setC(Double.parseDouble(getTagValue("c",element)));
				wave.setBrightness(Integer.parseInt(getTagValue("brightness",element)));
				wave.setAlpha(Integer.parseInt(getTagValue("alpha",element)));
				String[] points = getTagValue("points",element).split(",");
				double[][] doublepoints = new double[points.length][3];
				for(int n=0; n<points.length; n++){
					String[] vertex = points[n].split(":");
					doublepoints[n][0] = Double.parseDouble(vertex[0]);
					doublepoints[n][1] = Double.parseDouble(vertex[1]);
					doublepoints[n][2] = Double.parseDouble(vertex[2]);
				}
				wave.setPoints(doublepoints);
				sprites.put(spriteIndex, wave);
				waves.put(spriteIndex, wave);
				spriteIndex++;
			}
		}
	}
	
	 private static String getTagValue(String sTag, Element eElement){
		  NodeList nlList= eElement.getElementsByTagName(sTag).item(0).getChildNodes();
		  Node nValue = (Node) nlList.item(0); 
		  return nValue.getNodeValue(); 
	 }


	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	
	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			Iterator<Sprite> iter = sprites.values().iterator();
			while(iter.hasNext()){
				Sprite sprite = (Sprite)iter.next();
				if(mirror){
					((Wave)sprite).draw(r, brightness);
				} else {
					sprite.draw(r);
				}
			}
			c.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return false;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
		if(sprite instanceof Wave){
			waves.remove(sprite.getID());
		}
	}
	
	public ConcurrentHashMap<Integer,Wave> getWaves(){
		return waves;
	}
	
	public Wave getWave(int id){
		return waves.get(id);
	}
	
	public void addWave(int id, Wave wave){
		wave.addListener(this);
		sprites.put(id, wave);
		waves.put(id, wave);
	}
	
	public void setTint(int brightness){
		this.brightness = brightness;
	}
	
	public void mirror(){
		mirror = true;
	}
	
	
	
	
	
	
	/*
	
	// THIS IS ALL THE ORIGINAL 'ANIMATION' BASED WAVE SHOW 
	
	// solution of wave equation with damping and the FPU cubic nonlinearity
	static private final double PI = 3.14159265358979323846264338327950;
	private double Y[][] = new double[GRIDLENGTH][3];  // numerical grid
	private int prevT = 0, curT = 1, nextT = 2;
	private double dt = .1, dx = .02, c = .12, damp = 0., fpu = 0.;
	static private final int GRIDLENGTH = 174;//64;	// TODO should be equivalent to light width + gaps
	//static private final int WIDTH = 580, HEIGHT = 220;
	//static private final double xscale = WIDTH/GRIDLENGTH, yscale = HEIGHT/5.4, dampScale = 100., fpuScale = 100.;
	//static private final int xoffs = (int)(0.5 + xscale/2), yoffs = HEIGHT/2;
	static private final double MAXDAMP = 1., MAXFPU = 1.;	// for use with sliders
	static private int WIDTH, HEIGHT, xoffs, yoffs;
	static private double xscale, yscale;
	
	public WaveShow(Raster r){
		this.r = r;
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		WIDTH = c.width;
		HEIGHT = c.height;
		xscale = c.width/(float)(GRIDLENGTH-1);
		//yscale = HEIGHT/5.4;
		yscale = HEIGHT/3;
		xoffs = 0;
		yoffs = HEIGHT/2 + (HEIGHT/10);
		initshape();	// starts the initial wave motion
	}

	public Raster getFrame() {
		iterate();	// THIS IS WHERE THE MAGIC HAPPENS
		
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			
			int px, py, x, y;
			px = xoffs;
			py = (int)(Y[0][curT]*yscale + yoffs);
			for(int i=1; i<GRIDLENGTH; i++) {
				x = (int)(i*xscale) + xoffs;
				y = (int)(Y[i][curT]*yscale + yoffs);
				//c.stroke(255);
				//c.line(px, py, x, y);
				// top of rectangle is between x and px
				c.noStroke();
				c.fill(0,150,255,255);
				c.rect(px, py+((y-py)/2), x-px, c.height);
				px = x;
				py = y;
			}
			
			c.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return false;
	}
	
	
	
	
	

	// THESE ALLOW YOU TO SET PROPERTIES FROM THE CONTROL PANEL
	
	public void setDamping(double d){
		damp = d;
	}
	
	public void setNonlinearity(double nl){
		fpu = nl;
	}
	
	public void setYoffset(double yoffset){
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			yoffs = (int)(yoffset*c.height);
		}
	}
	
	public void setDX(double dx){	// TODO find out what this does
		this.dx = dx;
	}
	
	public void setC(double c){	// TODO find out what this does
		this.c = c;
	}
	
	
	

	 // THESE ARE ALL WAVE PHYSICS FUNCTIONS

	public void createImpact(float x, float y){
		// TODO this will be the function where a force is 
		// specified on the raster to create a new wave.
		
//		int i = (int)((x - xoffs)/xscale);
//		double a = (y - yoffs)/yscale;
//		if (0 < i && i < GRIDLENGTH-1)
//			Y[i][prevT] = Y[i][curT] = a;
		 
	}
	
	protected double sech(double x)
	{
		return 2./(Math.exp(x) + Math.exp(-x));
	}
	
	protected void initshape()
	{
		for (int m = 0; m < GRIDLENGTH; m++)
			Y[m][prevT] = Y[m][curT] = 2.5*sech((double)(m - GRIDLENGTH/2)/5.)* Math.sin(m*10.*PI/(GRIDLENGTH-1));
		Y[GRIDLENGTH-1][prevT] = Y[GRIDLENGTH-1][curT] = 
			Y[GRIDLENGTH-1][nextT] = Y[0][prevT] = Y[0][curT] =
				Y[0][nextT] = 0.;
	}
	
	// prepare grid for next time step
	protected void switchgrid()
	{
		prevT = (prevT+1) % 3;
		curT = (curT+1) % 3;
		nextT = (nextT+1) % 3;
	}

	// calculate string shape at time t+dt from shape at time t
	// simple explicit finite-difference method
	protected void iterate()
	{
		double t1, t2;
		for (int m=1; m<GRIDLENGTH-1; m++) {
			t1 = Y[m+1][curT] - Y[m][curT];
			t2 = Y[m][curT] - Y[m-1][curT];
			Y[m][nextT] = c*c*(dt/dx)*(dt/dx)*(Y[m-1][curT] - 2*Y[m][curT] +
					Y[m+1][curT] + fpu*(t1*t1 - t2*t2)) - Y[m][prevT] + 2*Y[m][curT] -
					damp*dt*(Y[m][curT] - Y[m][prevT]);
		}
		switchgrid();
	}
	
	*/

}
