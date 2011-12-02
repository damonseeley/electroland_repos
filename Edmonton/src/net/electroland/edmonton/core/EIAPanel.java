package net.electroland.edmonton.core;

/**
 * Draw the ELU canvas and various indicators in strips
 */

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

import javax.swing.JPanel;

public class EIAPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	private static final long serialVersionUID = 1L;
	//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
	final static float dash1[] = {10.0f};
	final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);
	final static Color bg = Color.white;
	final static Color fg = Color.black;
	final static Color red = Color.red;
	final static Color white = Color.white;

	private int width, height, cols, rows;
	private int margin;
	private int boxWidth;

	//constructor
	public EIAPanel (int w, int h) {

		this.width = w;
		this.height = h;
		setBackground(bg);
		setSize(w, h);
		setPreferredSize(new Dimension(w, h)); // need both SetSize and SetPreferredSize here for some reason

		addMouseMotionListener(this);

		margin = 32;
		rows = 2;
		cols = 4;
		boxWidth = (width - margin*2 - (margin/2)*(cols-1)) / cols;
		//int[] xLocs = new int[8];

	}

	public void paintSensors(boolean[] states, boolean[] changed){

		Graphics2D gci = (Graphics2D)this.getGraphics();
		gci.setColor(new Color(10,10,10));
		gci.fillRect(0,0,width,height);

		int i = 0;
		for (int r=0; r<rows; r++){
			for(int c=0; c<cols; c++) {
				if (states[i]) {
					//paint a red box
					if (changed[i] == true){
						//paint a red box
						gci.setColor(new Color(255,0,0));
						gci.fillRect(c*boxWidth+margin+c*margin/2,r*boxWidth+margin+r*margin/2,boxWidth,boxWidth);
					} else {
						//paint a white box
						gci.setColor(new Color(255,255,255));
						gci.fillRect(c*boxWidth+margin+c*margin/2,r*boxWidth+margin+r*margin/2,boxWidth,boxWidth);
					}
					
					i++;
				} else {
					//paint an outline
					gci.setColor(new Color(64,64,64));
					gci.drawRect(c*boxWidth+margin+c*margin/2,r*boxWidth+margin+r*margin/2,boxWidth,boxWidth);
					Font afont = new Font("afont",Font.PLAIN,24);
					gci.setFont(afont);
					gci.drawString((i + 1)+"", c*boxWidth+margin+c*margin/2+boxWidth/2, r*boxWidth+margin+r*margin/2+boxWidth/2);
					i++;
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
