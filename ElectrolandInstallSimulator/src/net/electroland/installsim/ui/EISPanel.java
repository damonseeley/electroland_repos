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
import net.electroland.installsim.core.InstallSimMainThread;
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
		
		
		final static float xScale = InstallSimMainThread.xScale;
		final static float yScale = xScale;
		final static float xOffset = InstallSimMainThread.xOffset;
		final static float yOffset = InstallSimMainThread.yOffset;
		
		public static int dummyPerson = InstallSimMainThread.DUMMYID;
		
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
			Enumeration<PhotoelectricTripWire> sensors = InstallSimMainThread.Sensors.elements();
			while(sensors.hasMoreElements()) {
				PhotoelectricTripWire s = sensors.nextElement();
				s.render(g2);
				
			}
			
			
			
			
			
			//draw people
			
			//fill the person circle
			Color c = new Color(255,0,0);
			
			float w = (float) (personCircle.width * 0.5f); // lets compute these values once per frame rather than once per person
			float h = (float) (personCircle.height * 0.5f);
			Enumeration<Person> persons = InstallSimMainThread.people.elements();
			while(persons.hasMoreElements()) {
				Person p = persons.nextElement();
				//System.out.println("person " + p.id + " (" + p.x + ", " + p.y + ")");
				personCircle.x = p.x * xScale - w + xOffset;
				personCircle.y = p.y * yScale - h + yOffset;
				
				g2.setColor(p.color);
				g2.fill(personCircle);
				
				c = new Color(128,128,128);
				g2.setColor(c);
				Font font = new Font("Arial", Font.PLAIN, 11);
			    g2.setFont(font);
				g2.drawString("ID: " + p.id.toString(), (int)personCircle.x-1, (int)personCircle.y-4);
				g2.drawString((int)p.x + ", " + (int)p.y + ", " + (int)p.z, (int)personCircle.x-10, (int)personCircle.y+h*3+1);
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
				p = InstallSimMainThread.people.get(InstallSimMainThread.DUMMYID);
				if (p!=null){
					p.setLoc((float)e.getX()-xOffset,(float)e.getY()-yOffset);
				}
		}



		public void mouseMoved(MouseEvent e) {
				// do nothing			
		}
		
		
		
	


}
