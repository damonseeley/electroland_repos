package net.electroland.laface.shows;

/*

vibrating string simulation

by Mariusz H. Jakubowski (mj@cs.princeton.edu)

started: 4/25/96

*/

import java.awt.*;
import java.applet.Applet;

public class Waveq extends java.applet.Applet implements Runnable
{
	// solution of wave equation with damping and the FPU cubic nonlinearity
	static final double PI = 3.14159265358979323846264338327950;
	double Y[][] = new double[GRIDLENGTH][3];  // numerical grid
	int prevT = 0, curT = 1, nextT = 2;
	double dt = .1, dx = .02, c = .04, damp = 0., fpu = 0.;
	static final int GRIDLENGTH = 64;
	static final int WIDTH = 580, HEIGHT = 220;
	static final double xscale = WIDTH/GRIDLENGTH, yscale = HEIGHT/5.4, dampScale = 100., fpuScale = 100.;
	static final int xoffs = (int)(0.5 + xscale/2), yoffs = HEIGHT/2;
	static final double MAXDAMP = 1., MAXFPU = 1.;

	// interface
	Thread waveqThread = null;
	int status;
	static final int running = 0, editing = 1;
	String statusText[] = new String[2];
	int delay = 10;
	Scrollbar dampingSlider, fpuSlider;

	// double buffering
	Dimension offDimension;
	Image offImage;
	Graphics offGraphics;

	// ** wave equation solution **

	// initialize string shape

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

	// ** user interface **

	// set up user interface components
	public void init()
	{
		resize(WIDTH, HEIGHT);
		setBackground(Color.black);
		setForeground(Color.white);
		initshape();
		statusText[running] = "Running string simulation";
		statusText[editing] = "Drag mouse to adjust string shape";
		status = running;
		showStatus(statusText[status]);
		// control interface
		// BorderLayout used to place a panel at the top of the applet,
		// GridBagLayout used to lay out the components inside the panel
		setLayout(new BorderLayout());
		Panel p = new Panel();
		GridBagLayout gridbag = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		c.weightx = 1.0;
		p.setLayout(gridbag);
		// lay out controls and add them to panel
		Button runButton = new Button("Run");
		gridbag.setConstraints(runButton, c);
		p.add(runButton);
		Button editButton = new Button("Edit");
		gridbag.setConstraints(editButton, c);
		p.add(editButton);
		p.add(new Label("Damping", Label.RIGHT));
		dampingSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0,(int)(MAXDAMP*dampScale));
		dampingSlider.setLineIncrement(1);
		dampingSlider.setForeground(Color.black);
		dampingSlider.setBackground(Color.white);
		c.fill = GridBagConstraints.BOTH;
		c.gridwidth = 3;
		gridbag.setConstraints(dampingSlider, c);
		p.add(dampingSlider);
		p.add(new Label("Nonlinearity", Label.RIGHT));
		fpuSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0,(int)(MAXFPU*fpuScale));
		fpuSlider.setLineIncrement(1);
		fpuSlider.setForeground(Color.black);
		fpuSlider.setBackground(Color.white);
		gridbag.setConstraints(fpuSlider, c);
		p.add(fpuSlider);
		// place panel at top of applet
		add("North", p);
	}

	// start main thread
	public void start()
	{
		if(waveqThread == null) {
			waveqThread = new Thread(this);
			waveqThread.start();
	  	}
	}

	// stop main thread
	public void stop()
	{
		waveqThread = null;
		offImage = null;
		offGraphics = null;
	}

	// main thread action
	public void run() {
		while (waveqThread != null) {
			try { Thread.sleep(delay); } catch (InterruptedException e) {}
			repaint();
			iterate();
		}
		waveqThread = null;
	}

	public void paint(Graphics g)
	{
		if (offImage != null)
			g.drawImage(offImage, 0, 0, null);
	}

	// update() method overridden to implement double buffering
	public void update(Graphics g)
	{
		int px, py, x, y;
		Dimension d = size();
		Color bg = getBackground(), fg = getForeground();
	  // create the offscreen graphics context
		if (offGraphics == null ||
				d.width != offDimension.width ||
				d.height != offDimension.height) {
			offDimension = d;
			offImage = createImage(d.width, d.height);
			offGraphics = offImage.getGraphics();
		}
		// erase the previous image
		offGraphics.setColor(bg);
		offGraphics.fillRect(0, 0, d.width, d.height);
		offGraphics.setColor(fg);
		// draw the next image
		px = xoffs;
		py = (int)(Y[0][curT]*yscale + yoffs);
		for(int i=1; i<GRIDLENGTH; i++) {
			x = (int)(i*xscale) + xoffs;
			y = (int)(Y[i][curT]*yscale + yoffs);
			offGraphics.drawLine(px, py, x, y);
			px = x;
			py = y;
		}
		// paint the image on the screen
		g.drawImage(offImage, 0, 0, null);
	}

	// called when mouse is dragged
	// allows user to adjust string shape whether or not string is in motion
	public boolean mouseDrag(Event evt, int x, int y)
	{
		int i = (int)((x - xoffs)/xscale);
		double a = (y - yoffs)/yscale;
		if (0 < i && i < GRIDLENGTH-1)
			Y[i][prevT] = Y[i][curT] = a;
		if (status == editing) repaint();
		return true;
	}

	// called when mouse exits applet space
	public boolean mouseExit(Event evt, int x, int y)
	{
		showStatus("");
		return true;
	}

	// called when mouse enters applet space
	public boolean mouseEnter(Event evt, int x, int y)
	{
		showStatus(statusText[status]);
		return true;
	}

	// event handler to respond to user's actions on buttons
	public boolean action(Event evt, Object arg)
	{
		if (evt.target instanceof Button) {
			String command = (String)arg;
			if (command.equals("Run")) {
				if (status != running) { 
					status = running; waveqThread.resume(); 
				}
			}
			else if (command.equals("Edit")) {
				if (status != editing) { 
					status = editing; waveqThread.suspend(); 
				}
			}
		}
		showStatus(statusText[status]);
		return true;
	}

	// low-level event handler to respond to user's actions on sliders
	public boolean handleEvent(Event evt)
	{
		if (evt.target instanceof Scrollbar)
			if (evt.target.equals(dampingSlider))
				damp = dampingSlider.getValue()/dampScale;
			else if (evt.target.equals(fpuSlider))
				fpu = fpuSlider.getValue()/fpuScale;
		return super.handleEvent(evt);
	}
}