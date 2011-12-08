package net.electroland.edmonton.core;

/**
 * Draw the ELU canvas and various indicators in strips
 */

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.Hashtable;

import javax.swing.JPanel;
import javax.vecmath.Point3d;

import net.electroland.ea.AnimationManager;
import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

	//some static stuff here for drawing
	private static final long serialVersionUID = 1L;
	//private static BasicStroke stroke = new BasicStroke(2.0f); Never used EGM
	final static float dash1[] = {10.0f};
	final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);


	public Hashtable<String, Object> context;
	private ELUCanvas2D canvas;
	private ELUManager elu;
	private IOManager eio;
	private ElectrolandProperties props;
	private double displayScale;
	private boolean showGraphics;
	private AnimationManager anim;

	//panel dims and margin info
	private int width, height;
	private int margin;
	private int stateSize;
	private int lightHeight, lightWidth;
	private double p1x,p1y,p1width,p1height,p2x,p2y,p2width,p2height;

	static Logger logger = Logger.getLogger(EIAFrame.class);
	
	//constructor
	public EIAPanel (Hashtable context) {

		this.context = context;
		
		addMouseMotionListener(this);


		this.elu = (ELUManager)context.get("elu");
		this.eio = (IOManager)context.get("eio");
		this.canvas = (ELUCanvas2D)context.get("canvas");
		this.props = (ElectrolandProperties)context.get("props");
		try {
			this.displayScale = props.getOptionalDouble("settings", "global", "displayScale");
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			displayScale = 1.0;
		}
		try {
			this.margin = props.getOptionalInt("settings", "global", "margin");
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			margin = 32;
		}
		try {
			showGraphics = Boolean.parseBoolean(props.getOptional("settings", "global", "showGraphics"));
		} catch (OptionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			showGraphics = true;
		}

		try {
			stateSize = props.getOptionalInt("settings", "global", "IStateSize");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			stateSize = 10;
		}

		try {
			lightHeight = props.getOptionalInt("settings", "global", "lightHeight");
			lightWidth = props.getOptionalInt("settings", "global", "lightWidth");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			lightHeight = 2;
			lightWidth = 2;
		}
		try {
			p1x = props.getOptionalDouble("peoplemover", "p1", "x");
			p1y = props.getOptionalDouble("peoplemover", "p1", "y");
			p1width = props.getOptionalDouble("peoplemover", "p1", "width");
			p1height = props.getOptionalDouble("peoplemover", "p1", "height");
			p2x = props.getOptionalDouble("peoplemover", "p2", "x");
			p2y = props.getOptionalDouble("peoplemover", "p2", "y");
			p2width = props.getOptionalDouble("peoplemover", "p2", "width");
			p2height = props.getOptionalDouble("peoplemover", "p2", "height");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			p1x = 0;
			p1y = 0;
			p1width = 0;
			p1height = 0;
			p2x = 0;
			p2y = 0;
			p2width = 0;
			p2height = 0;
		}
		
		anim = (AnimationManager)context.get("anim");
		
		
		// SWING
		
		//eventually need to update height here for tiling
		this.width = (int)(canvas.getDimensions().width*displayScale + (margin*2));
		this.height = (int)(canvas.getDimensions().height*displayScale + (margin*2));
		logger.info("EIAPanel dims: " + width + " " + height);
		
		setBackground(Color.BLUE);
		this.setSize(width, height);
		setPreferredSize(new Dimension(width, height)); // need both SetSize and SetPreferredSize here for some reason
		
		
		logger.info("EIAPanel loaded with displayScale of " + displayScale);

	}


	public void update(){
		//called from Conductor
		repaint();

	}



	public void paint(Graphics g) {
		if (showGraphics) {

			/*
			 * While it is painful do ALL displayScale calculations here to 
			 * avoid confusion over where scaling occurs
			 */

			//clear image from last time
			clear(g);

			// Create BI on which to draw everything double-buffered, add the margin to total width and height
			int biWidth = (int)((canvas.getDimensions().width+margin)*displayScale);
			int biHeight = (int)((canvas.getDimensions().height+margin)*displayScale);
			BufferedImage bi = new BufferedImage(biWidth, biHeight, BufferedImage.TYPE_INT_RGB);		

			Graphics2D g2 = (Graphics2D)bi.getGraphics();
			//set styles
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			//test
			//g2.fillRect(50,50,100,100);
			
			
			

			
			

			/*
			 * Draw Anim/Canvas
			 */
			// Get image from AnimationManager
			//NEED TO SCALE HERE
			Dimension d = anim.getStageDimensions();
			//g2.drawImage(anim.getStage(),0,0,null);
			g2.drawImage(anim.getStage(),0,0,(int)(d.width*displayScale),(int)(d.height*displayScale),null);
			
			//outline the canvas
			g2.setColor(new Color(48, 32, 48));
			g2.drawRect(0,0,(int)(canvas.getDimensions().width*displayScale),(int)(canvas.getDimensions().height*displayScale));

			
			/*
			 * Draw people mover and other static elements
			 */
			g2.setColor(new Color(32, 64, 32));
			g2.drawRect((int)(p1x*displayScale), (int)(p1y*displayScale),(int)(p1width*displayScale),(int)(p1height*displayScale));
			g2.drawRect((int)(p2x*displayScale), (int)(p2y*displayScale),(int)(p2width*displayScale),(int)(p2height*displayScale));
			


			



			/*
			 * Draw Sensors
			 */
			for (IOState state : eio.getStates())
			{
				Point3d l = (Point3d) state.getLocation().clone();
				l.scale(displayScale);
				// render sprite

				int brite = 0;
				IState is = (IState)state;
				if (is.getState()) {
					brite = 255;
				}

				//draw the sensor state
				g2.setColor(new Color(brite, brite, brite));
				g2.fillRect((int)(l.x)-(stateSize/2), (int)(l.y)-(stateSize/2),stateSize, stateSize);
				//draw an outline
				g2.setColor(new Color(64, 64, 64));
				g2.drawRect((int)(l.x)-(stateSize/2), (int)(l.y)-(stateSize/2),stateSize, stateSize);

			}


			/*
			 * Draw light fixtures
			 */		

			for (Fixture fix : elu.getFixtures())
			{
				Point3d l = (Point3d)fix.getLocation().clone();
				//logger.info("orig " + (int)l.x + " " + (int)l.y);
				l.scale(displayScale);
				g2.setColor(new Color(0, 0, 196));
				g2.drawRect((int)(l.x)-lightWidth/2, (int)(l.y)-lightHeight/2, lightWidth, lightHeight);
			}


			/*
			 * Finally, draw it all on the Panel
			 */
			g.setColor(Color.BLACK);
			g.fillRect(0,0,bi.getWidth()+margin,bi.getHeight()+margin);
			g.drawImage(bi, margin, margin, null);

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
		//System.out.println(e);
	}

	public void mouseMoved(MouseEvent e) {
		// do nothing			
	}






}
