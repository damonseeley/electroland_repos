package net.electroland.skate.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

import javax.swing.JPanel;

public class GUIPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	private static final long serialVersionUID = 1L;
		//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
		final static float dash1[] = {10.0f};
		final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);
		final static Color bg = Color.white;
		final static Color fg = Color.black;
		final static Color red = Color.red;
		final static Color white = Color.white;
		
		/*
		final static float xScale = InstallSimMain.xScale;
		final static float yScale = xScale;
		final static float xOffset = InstallSimMain.xOffset;
		final static float yOffset = InstallSimMain.yOffset;
		*/
		
		//public static int dummyPerson = InstallSimMain.DUMMYID;
		
//		Light[] lights;  Don't dup variable for no reason EGM
//		ConcurrentHashMap<Integer, Person>  people;
		
		//example vars
		//private Ellipse2D.Double personCircle = new Ellipse2D.Double(10, 10, 20 * xScale, 20 * yScale);

		//constructor
		public GUIPanel (int w, int h) {

		    setBackground(bg);
		    setSize(w, h);
		    setPreferredSize(new Dimension(w, h)); // need both SetSize and SetPreferredSize here for some reason

		    addMouseMotionListener(this);
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
