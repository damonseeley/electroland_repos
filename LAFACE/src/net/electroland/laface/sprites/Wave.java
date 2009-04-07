package net.electroland.laface.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.laface.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Wave extends Sprite {
	
	// solution of wave equation with damping and the FPU cubic nonlinearity
	static private final double PI = 3.14159265358979323846264338327950;
	private double Y[][] = new double[GRIDLENGTH][3];  // numerical grid
	private int prevT = 0, curT = 1, nextT = 2;
	private double dt = .1, dx = .02, c = .08, damp = 0., fpu = 0.;
	static private final int GRIDLENGTH = 64;	// TODO should be equivalent to light width + gaps
	static private final double MAXDAMP = 1., MAXFPU = 1.;	// for use with sliders
	private int WIDTH, HEIGHT, xoffs, yoffs;
	static private double xscale, yscale;
	private int brightness, alpha;

	public Wave(int id, Raster raster, float x, float y) {
		super(id, raster, x, y);
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			WIDTH = c.width;
			HEIGHT = c.height;
			xscale = c.width/(float)(GRIDLENGTH-1);
		}
		yscale = HEIGHT/3;
		xoffs = 0;
		yoffs = HEIGHT/2 + (HEIGHT/10);
		brightness = 255;
		alpha = 255;
		initshape();	// starts the initial wave motion
	}

	@Override
	public void draw(Raster r) {
		iterate();	// THIS IS WHERE THE MAGIC HAPPENS
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.noStroke();
			c.fill(brightness,brightness,brightness,alpha);
			c.beginShape();
			int lowest = c.height;	// lowest point (highest value) in wave
			int px, py, x, y;
			px = xoffs;
			py = (int)(Y[0][curT]*yscale + yoffs);
			c.vertex(px,py);
			for(int i=1; i<GRIDLENGTH; i++) {
				x = (int)(i*xscale) + xoffs;
				y = (int)(Y[i][curT]*yscale + yoffs);
				//c.rect(px, py+((y-py)/2), x-px, c.height-(py+((y-py)/2)));	// vertical bar for each point
				c.vertex(x,y);
				if(y > lowest){
					lowest = y;
				}
				px = x;
				py = y;
			}
			c.vertex(c.width,lowest);
			c.vertex(0,lowest);
			c.endShape(PConstants.CLOSE);
		}
	}
	
	public void draw(Raster r, int brightnessValue) {
		// THIS DOES NOT ITERATE OVER THE WAVE FUNCTION, SIMPLY DRAWS CURRENT STATE
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.noStroke();
			c.fill(brightnessValue,brightnessValue,brightnessValue,alpha);
			c.beginShape();
			int lowest = c.height;	// lowest point (highest value) in wave
			int px, py, x, y;
			px = xoffs;
			py = (int)(Y[0][curT]*yscale + yoffs);
			c.vertex(px,py);
			for(int i=1; i<GRIDLENGTH; i++) {
				x = (int)(i*xscale) + xoffs;
				y = (int)(Y[i][curT]*yscale + yoffs);
				//c.rect(px, py+((y-py)/2), x-px, c.height-(py+((y-py)/2)));	// vertical bar for each point
				c.vertex(x,y);
				if(y > lowest){
					lowest = y;
				}
				px = x;
				py = y;
			}
			c.vertex(c.width,lowest);
			c.vertex(0,lowest);
			c.endShape(PConstants.CLOSE);
		}
	}
	
	
	
	
	
	
	
	
	// ALLOW CONTROL PANEL TO SET PROPERTIES
	
	public void setAlpha(int alpha){
		if(alpha >= 0 && alpha <= 255){
			this.alpha = alpha;
		}
	}
	
	public void setBrightness(int brightness){
		if(brightness >= 0 && brightness <= 255){
			this.brightness = brightness;
		}
	}
	
	public void setDamping(double d){
		damp = d;
	}
	
	public void setNonlinearity(double nl){
		fpu = nl;
	}
	
	public void setYoffset(double yoffset){
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			yoffs = (int)(yoffset*c.height);
		}
	}
	
	public void setDX(double dx){	// TODO find out what this does
		this.dx = dx;
	}
	
	public void setC(double c){	// TODO find out what this does
		this.c = c;
	}
	
	public void setPoints(double[][] points){
		Y = points;
	}
	
	public void reset(){
		initshape();
	}
	
	
	
	
	
	
	// ALLOWS CONTROL PANEL TO READ PROPERTIES

	public int getAlpha(){
		return alpha;
	}
	
	public int getBrightness(){
		return brightness;
	}
	
	public double getDamping(){
		return damp;
	}
	
	public double getNonlinearity(){
		return fpu;
	}
	
	public double getYoffset(){
		return (double)yoffs/HEIGHT;
	}
	
	public double getDX(){
		return dx;
	}
	
	public double getC(){
		return c;
	}
	
	public double[][] getPoints(){
		return Y;
	}
	
	
	
	

	// THESE ARE ALL WAVE PHYSICS FUNCTIONS

	public void createImpact(float x, float y){
		// TODO this will be the function where a force is 
		// specified on the raster to create a new wave.
		
		int i = (int)((x - xoffs)/xscale);
		double a = (y - yoffs)/yscale;
		if (0 < i && i < GRIDLENGTH-1)
			Y[i][prevT] = Y[i][curT] = a;
		 
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
