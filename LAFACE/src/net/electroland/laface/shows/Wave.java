package net.electroland.laface.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Wave implements Animation {
	
	private Raster r;
	
	// solution of wave equation with damping and the FPU cubic nonlinearity
	static private final double PI = 3.14159265358979323846264338327950;
	private double Y[][] = new double[GRIDLENGTH][3];  // numerical grid
	private int prevT = 0, curT = 1, nextT = 2;
	private double dt = .1, dx = .02, c = .04, damp = 0., fpu = 0.;
	static private final int GRIDLENGTH = 64;
	//static private final int WIDTH = 580, HEIGHT = 220;
	//static private final double xscale = WIDTH/GRIDLENGTH, yscale = HEIGHT/5.4, dampScale = 100., fpuScale = 100.;
	//static private final int xoffs = (int)(0.5 + xscale/2), yoffs = HEIGHT/2;
	static private final double MAXDAMP = 1., MAXFPU = 1.;	// for use with sliders
	static private int WIDTH, HEIGHT, xoffs, yoffs;
	static private double xscale, yscale, dampScale = 100. ,fpuScale = 100.;	// for use with sliders
	
	public Wave(Raster r){
		this.r = r;
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		WIDTH = c.width;
		HEIGHT = c.height;
		xscale = WIDTH/GRIDLENGTH;
		yscale = HEIGHT/5.4;
		xoffs = 0;
		yoffs = HEIGHT/2;
		initshape();
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
	
	
	
	/**
	 * THESE ARE ALL WAVE PHYSICS FUNCTIONS
	 */
	
	public void createImpact(float x, float y){
		// TODO this will be the function where a force is 
		// specified on the raster to create a new wave.
		/*
		int i = (int)((x - xoffs)/xscale);
		double a = (y - yoffs)/yscale;
		if (0 < i && i < GRIDLENGTH-1)
			Y[i][prevT] = Y[i][curT] = a;
		 */
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

}
