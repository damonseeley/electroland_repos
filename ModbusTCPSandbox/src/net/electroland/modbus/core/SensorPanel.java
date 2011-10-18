package net.electroland.modbus.core;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

import javax.swing.JPanel;

public class SensorPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	private static final long serialVersionUID = 1L;
	//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
	final static float dash1[] = {10.0f};
	final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);
	final static Color bg = Color.white;
	final static Color fg = Color.black;
	final static Color red = Color.red;
	final static Color white = Color.white;

	private int width, height;
	private int margin;
	private int boxWidth;

	//constructor
	public SensorPanel (int w, int h) {

		this.width = w;
		this.height = h;
		setBackground(bg);
		setSize(w, h);
		setPreferredSize(new Dimension(w, h)); // need both SetSize and SetPreferredSize here for some reason

		addMouseMotionListener(this);
		
		margin = 32;
		boxWidth = (width - margin - (margin/2)*3) / 4;
		//int[] xLocs = new int[8];
		
	}

	public void paintSensors(boolean[] states, int lastInput){
		
		Graphics2D gci = (Graphics2D)this.getGraphics();
		gci.setColor(new Color(10,10,10));
		gci.fillRect(0,0,width,height);
		
		for (int i=0; i<states.length; i++){
			int yloc = margin/2;
			if (i > 4){
				yloc = boxWidth + margin;
			}
			if (states[i]) {
				//paint a solid box
				if (lastInput == i){
					//paint a red box
					gci.setColor(new Color(128,0,0));
					gci.fillRect(i*boxWidth+margin+i*margin/2,yloc,boxWidth,boxWidth);
				} else {
					//paint a white box
					gci.setColor(new Color(256,256,256));
					gci.fillRect(i%4,yloc,boxWidth,boxWidth);
				}
			}
		}
		
		
		

		
		
	}



	public void paint(Graphics g) {
		clear(g);
		Graphics2D g2 = (Graphics2D)g;

		//set styles
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		//g2.setStroke(dashed);

		/*
			//draw sensors
			Enumeration<Sensor> sensors = InstallSimMain.sensors.elements();
			while(sensors.hasMoreElements()) {
				PhotoelectricTripWire s = (PhotoelectricTripWire)sensors.nextElement();
				s.render(g2);

			}

			//draw people
			Enumeration<Person> persons = InstallSimMain.people.elements();
			while(persons.hasMoreElements()) {
				Person p = persons.nextElement();
				p.render(g2, p.id);

			}
		 */


	}


	// super.paintComponent clears offscreen pixmap,
	// since we're using double buffering by default.

	protected void clear(Graphics g) {
		super.paintComponent(g);
	}


	public void mouseEntered(MouseEvent e) {
		//System.out.println(e);
	}

	public void mouseExited(MouseEvent e) {
		//System.out.println(e);
	}

	public void mouseClicked(MouseEvent e) {
		//System.out.println(e);
	}


	public void mouseDragged(MouseEvent e) {
		//System.out.println(e);
	}

	public void mouseMoved(MouseEvent e) {
		// do nothing			
	}






}
