package net.electroland.installsim.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Ellipse2D;
import java.util.Enumeration;

import javax.swing.JPanel;

import org.apache.log4j.Logger;

import net.electroland.installsim.core.InstallSimEIA;
import net.electroland.installsim.core.InstallSimMemphis;
import net.electroland.installsim.core.ModelEIA;
import net.electroland.installsim.core.ModelGeneric;
import net.electroland.installsim.core.Person;
import net.electroland.installsim.sensors.PhotoelectricTripWire;
import net.electroland.installsim.sensors.Sensor;

public class EISPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	static Logger logger = Logger.getLogger(EISPanel.class);

	private static final long serialVersionUID = 1L;
	//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
	final static float dash1[] = {10.0f};
	final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);
	final static Color bg = Color.white;
	final static Color fg = Color.black;
	final static Color red = Color.red;
	final static Color white = Color.white;

	private static float xScale;
	private static float yScale = xScale;
	private static float xOffset;
	private static float yOffset;

	private static int dummyPerson;

	private Ellipse2D.Double personCircle = new Ellipse2D.Double(10, 10, 20 * xScale, 20 * yScale);
	private ModelGeneric model;

	//constructor
	public EISPanel (int w, int h, ModelGeneric m) {

		model = m;
		setBackground(bg);
		setPreferredSize(new Dimension(w,h)); // you want to use setPreferredSize (setSize() doesn't work correctly I don't think) EGM
		
		xScale = model.xScale;
		yScale = xScale;
		xOffset = model.xOffset;
		yOffset = model.yOffset;

		dummyPerson = model.DUMMYID;

		addMouseMotionListener(this);
	}



	public void paint(Graphics g) {
		
		clear(g);
		if (model != null){
			model.render(g);
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
		//Person p;
		//p = model.people.get(dummyPerson);
		//if (p!=null){
			//p.setLoc((float)e.getX()-xOffset,(float)e.getY()-yOffset);
		//}
	}



	public void mouseMoved(MouseEvent e) {
		// do nothing			
	}






}
