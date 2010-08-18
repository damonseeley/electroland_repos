package net.electroland.installsim.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.Enumeration;

import javax.swing.JPanel;

import net.electroland.coopLights.core.*;
//import net.electroland.installsim.core.InstallSimConductor;
import net.electroland.installsim.core.InstallSimMain;
import net.electroland.installsim.core.Person;
import net.electroland.installsim.sensors.*;

public class EISPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	private static final long serialVersionUID = 1L;
		//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
		final static float dash1[] = {10.0f};
		final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);
		final static Color bg = Color.white;
		final static Color fg = Color.black;
		final static Color red = Color.red;
		final static Color white = Color.white;
		
		
		final static float xScale = InstallSimMain.xScale;
		final static float yScale = xScale;
		final static float xOffset = InstallSimMain.xOffset;
		final static float yOffset = InstallSimMain.yOffset;
		
		public static int dummyPerson = InstallSimMain.DUMMYID;
		
//		Light[] lights;  Don't dup variable for no reason EGM
//		ConcurrentHashMap<Integer, Person>  people;
		
		//example vars
		private Ellipse2D.Double personCircle = new Ellipse2D.Double(10, 10, 20 * xScale, 20 * yScale);

		//constructor
		public EISPanel () {

		    setBackground(bg);
		    setPreferredSize(new Dimension(800, 800)); // you want to use setPreferredSize (setSize() doesn't work correctly I don't think) EGM

		    addMouseMotionListener(this);
		}
		
	
		
		public void paint(Graphics g) {
			clear(g);
			Graphics2D g2 = (Graphics2D)g;
			
			//set styles
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			//g2.setStroke(dashed);

			
			//draw sensors
			Enumeration<Sensor> sensors = InstallSimMain.sensors.elements();
			while(sensors.hasMoreElements()) {
				PhotoelectricTripWire s = (PhotoelectricTripWire)sensors.nextElement();
				s.render(g2);
				
			}
			
			//draw spawn locations and other indicators
			InstallSimMain.god.render(g2);
			
			
			//draw people
			Enumeration<Person> persons = InstallSimMain.people.elements();
			while(persons.hasMoreElements()) {
				Person p = persons.nextElement();
				p.render(g2, p.id);
				
			}
			

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
				Person p;
				p = InstallSimMain.people.get(InstallSimMain.DUMMYID);
				if (p!=null){
					p.setLoc((float)e.getX()-xOffset,(float)e.getY()-yOffset);
				}
		}



		public void mouseMoved(MouseEvent e) {
				// do nothing			
		}
		
		
		
	


}
