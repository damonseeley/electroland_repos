package net.electroland.edmonton.core.ui;

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
import java.awt.image.BufferedImage;
import java.util.Map;

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

public class EIATiledPanel extends JPanel implements MouseMotionListener { // change mouselistener to mousemotionlistener to allow smooth drags

    //some static stuff here for drawing
    private static final long serialVersionUID = 1L;
    final static float dash1[] = {10.0f};
    final static BasicStroke dashed = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash1, 0.0f);

    private ELUCanvas2D canvas;
    private ELUManager elu;
    private IOManager eio;
    private ElectrolandProperties props, propsStatic;
    private double displayScale;
    private boolean showGraphics;
    private AnimationManager anim;
    private boolean showAnimation;

    //panel dims and margin info
    public int panelTileWidth;
    public int calcWidth, calcHeight;
    private int margin;
    private int intMargin;
    private int stateSize;
    private int lightHeight, lightWidth;
    private double p1x,p1y,p1width,p1height,p2x,p2y,p2width,p2height;
    private double s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11;

    static Logger logger = Logger.getLogger(EIAFrame.class);

    //constructor
    public EIATiledPanel (Map<String,Object> context) {

        addMouseMotionListener(this);

        this.anim = (AnimationManager)context.get("anim");
        this.elu = (ELUManager)context.get("elu");
        this.eio = (IOManager)context.get("eio");
        this.canvas = (ELUCanvas2D)context.get("canvas");
        this.props = (ElectrolandProperties)context.get("props");
        this.propsStatic = (ElectrolandProperties)context.get("propsStatic");


        try {
            showGraphics = Boolean.parseBoolean(props.getOptional("settings", "display", "showGraphics"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            showGraphics = true;
        }
        try {
            showAnimation = Boolean.parseBoolean(props.getOptional("settings", "display", "showAnimation"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            showAnimation = true;
        }
        try {
            this.displayScale = props.getOptionalDouble("settings", "displaymetrics", "displayScale");
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            displayScale = 1.0;
        }
        try {
            this.margin = props.getOptionalInt("settings", "displaymetrics", "margin");
            intMargin = margin/2;
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            margin = 32;
            intMargin = margin/2;
        }

        try {
            stateSize = props.getOptionalInt("settings", "displaymetrics", "IStateSize");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            stateSize = 10;
        }

        try {
            lightHeight = props.getOptionalInt("settings", "displaymetrics", "lightHeight");
            lightWidth = props.getOptionalInt("settings", "displaymetrics", "lightWidth");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            lightHeight = 2;
            lightWidth = 2;
        }
        try {
            p1x      = propsStatic.getOptionalDouble("peoplemover", "p1", "x");
            p1y      = propsStatic.getOptionalDouble("peoplemover", "p1", "y");
            p1width  = propsStatic.getOptionalDouble("peoplemover", "p1", "width");
            p1height = propsStatic.getOptionalDouble("peoplemover", "p1", "height");
            p2x      = propsStatic.getOptionalDouble("peoplemover", "p2", "x");
            p2y      = propsStatic.getOptionalDouble("peoplemover", "p2", "y");
            p2width  = propsStatic.getOptionalDouble("peoplemover", "p2", "width");
            p2height = propsStatic.getOptionalDouble("peoplemover", "p2", "height");
            
            s1  = propsStatic.getOptionalDouble("speaker", "s1", "x");
            s2  = propsStatic.getOptionalDouble("speaker", "s2", "x");
            s3  = propsStatic.getOptionalDouble("speaker", "s3", "x");
            s4  = propsStatic.getOptionalDouble("speaker", "s4", "x");
            s5  = propsStatic.getOptionalDouble("speaker", "s5", "x");
            s6  = propsStatic.getOptionalDouble("speaker", "s6", "x");
            s7  = propsStatic.getOptionalDouble("speaker", "s7", "x");
            s8  = propsStatic.getOptionalDouble("speaker", "s8", "x");
            s9  = propsStatic.getOptionalDouble("speaker", "s9", "x");
            s10 = propsStatic.getOptionalDouble("speaker", "s10", "x");
            s11 = propsStatic.getOptionalDouble("speaker", "s11", "x");

            
            
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
            
            s1 = 0;
            s2 = 0;
            s3 = 0;
            s4 = 0;
            s5 = 0;
            s6 = 0;
            s7 = 0;
            s8 = 0;
            s9 = 0;
            s10 = 0;
            s11 = 0;
            
        }

        setDisplayScale(displayScale);

    }

    public void setDisplayScale(double ds) {
        this.displayScale = ds;
        //eventually need to update height here for tiling
        this.calcWidth = (int)(canvas.getDimensions().width*displayScale + (margin*2) + (intMargin*2));
        this.calcHeight = (int)((canvas.getDimensions().height*displayScale)) + (margin*2) + (intMargin*2);

        setBackground(Color.BLUE);
        //this.setSize(calcWidth, calcHeight);
        try {
            panelTileWidth = props.getRequiredInt("settings", "global", "guiwidth");;
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            panelTileWidth = 1024;
            e.printStackTrace();
        }
        vTiles = (int)(Math.ceil((double)calcWidth/panelTileWidth));
        //logger.info(vTiles);

        this.setSize(panelTileWidth, calcHeight*vTiles);
        setPreferredSize(new Dimension(panelTileWidth, calcHeight*vTiles)); // need both SetSize and SetPreferredSize here for some reason

        logger.info("EIAPanel rescaled with displayScale of " + displayScale);
    }

    public int getPanelWidth() {
        return panelTileWidth;
    }

    public double getDisplayScale() {
        return displayScale;
    }

    public void update(){
        //called from Conductor
        repaint();
    }

    int vTiles;

    public void paint(Graphics g) {
        if (showGraphics) {

            /*
             * While it is painful do ALL displayScale calculations here to 
             * avoid confusion over where scaling occurs
             */

            //clear image from last time
            clear(g);

            // Create BI on which to draw everything double-buffered, add the int margin to total width and height
            int biHeight = (int)((canvas.getDimensions().height*displayScale)+(intMargin*2));
            //make it exactly three times the panel width for ease later
            BufferedImage bi = new BufferedImage(panelTileWidth*vTiles+(intMargin*2), biHeight, BufferedImage.TYPE_INT_RGB);    // maybe don't need to account for intMargin in width here???    

            Graphics2D g2 = (Graphics2D)bi.getGraphics();
            //set styles
            //g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            /*
             * Draw Anim/Canvas
             */
            if (showAnimation){
                // Get image from AnimationManager
                Dimension d = anim.getStageDimensions();
                g2.drawImage(anim.getStage(),intMargin,intMargin,(int)(d.width*displayScale),(int)(d.height*displayScale),null);

                //outline the canvas
                g2.setColor(new Color(48, 32, 48));
                g2.drawRect(intMargin,intMargin,(int)(canvas.getDimensions().width*displayScale),(int)(canvas.getDimensions().height*displayScale));
            }


            /*
             * Draw people mover and other static elements
             */
            g2.setColor(new Color(32, 64, 32));
            g2.drawRect((int)(p1x*displayScale)+intMargin, (int)(p1y*displayScale)+intMargin,(int)(p1width*displayScale),(int)(p1height*displayScale));
            g2.drawRect((int)(p2x*displayScale)+intMargin, (int)(p2y*displayScale)+intMargin,(int)(p2width*displayScale),(int)(p2height*displayScale));
            // speakers
            g2.setColor(new Color(32, 196, 32));
            double sWidth = 1.0;
            double sY = 1.0;
            double sOffset = 1.3;
            Font font = new Font("Arial", Font.PLAIN, 10);
    	    g2.setFont(font);
            g2.drawRect((int)(s1*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("1",(int)((s1+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s2*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("2",(int)((s2+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s3*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("3",(int)((s3+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s4*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("4",(int)((s4+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s5*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("5",(int)((s5+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s6*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("6",(int)((s6+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s7*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("7",(int)((s7+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s8*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("8",(int)((s8+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s9*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("9",(int)((s9+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s10*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("10",(int)((s10+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            g2.drawRect((int)(s11*displayScale)+intMargin, (int)(sY*displayScale)+intMargin,(int)(sWidth*displayScale),(int)(sWidth*displayScale));
    		g2.drawString("11",(int)((s11+sOffset)*displayScale+intMargin), (int)((sY+sOffset)*displayScale+intMargin));
            

            /*
             * Draw Sensors
             */
            Font font2 = new Font("Arial", Font.PLAIN, 10);
            g2.setFont(font2);
            double isOffset = 14.0;
            for (IOState state : eio.getStates())
            {
                Point3d l = (Point3d) state.getLocation().clone();
                l.scale(displayScale);
                // render sprite

                Color sColor = new Color(0, 0, 0);
                IState is = (IState)state;
                if (is.isSuspect()){
                    if (is.getState()){
                        sColor = new Color(255, 0, 0);
                    }else{
                        //SHOULD BE IMPOSSIBLE
                        sColor = new Color(0, 255, 0);
                    }
                }else{
                    if (is.getState()){
                        //sensor is ON and functioning correctly
                        sColor = new Color(255, 255, 255);
                    }else{
                        //sensor is OFF and functioning correctly
                        sColor = new Color(20, 20, 20);
                    }
                }

                //draw the sensor state
                g2.setColor(sColor);
                g2.fillRect((int)(l.x)-(stateSize/2)+intMargin, (int)(l.y)-(stateSize/2)+intMargin,stateSize, stateSize);
                //draw an outline
                g2.setColor(new Color(64, 64, 64));
                g2.drawRect((int)(l.x)-(stateSize/2)+intMargin, (int)(l.y)-(stateSize/2)+intMargin,stateSize, stateSize);
                
                //draw a label
                g2.setColor(new Color(96, 96, 96));
                g2.drawString(state.getID().toString(),(int)((l.x-6.0)+intMargin), (int)((l.y+isOffset)+intMargin));                
            }

            /*
             * Draw light fixtures
             */

            for (Fixture fix : elu.getFixtures())
            {
                Point3d l = (Point3d)fix.getLocation().clone();
                //logger.info("fixture " + fix.getName());
                l.scale(displayScale);
                g2.setColor(new Color(0, 0, 196));
                g2.drawRect((int)(l.x)-lightWidth/2+intMargin, (int)(l.y)-lightHeight/2+intMargin, lightWidth, lightHeight);
            }


            /*
             * Finally, draw it all on the Panel
             */

            int tileHeight = bi.getHeight()+margin*2;
            int tileWidth = panelTileWidth;

            //create a bufferedimage that is the width and height of the final tiled presentation
            BufferedImage bi2 = new BufferedImage(panelTileWidth*vTiles,tileHeight*vTiles,BufferedImage.TYPE_INT_RGB);
            Graphics2D g2b = (Graphics2D)bi2.getGraphics();    

            g2b.setColor(Color.BLACK);

            //logger.info("Tile Renders: " + vTiles);
            for (int i=0; i<vTiles; i++){
                g2b.drawImage(bi, 0, tileHeight*i, tileWidth, tileHeight*(i+1), tileWidth*i, 0, tileWidth*(i+1), tileHeight, null);
                //logger.info(0 + "    " + tileHeight*i + "    " + tileWidth+ "    " + tileHeight*(i+1)+ "    " + tileWidth*i+ "    " + 0 + "    " + tileWidth*(i+1)+ "    " + tileHeight);
            }

            // finally, draw the tile image on the Panel image
            g.setColor(Color.BLACK);
            g.fillRect(0,0,g.getClipBounds().width,g.getClipBounds().height);
            g.drawImage(bi2, margin, margin, null);
            //g2.dispose();
        } else {
        	//clear image from last time
            clear(g);
            //g.setColor(Color.BLACK);
            //g.fillRect(0,0,g.getClipBounds().width,g.getClipBounds().height);
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

	public boolean showHideGfx() {
		// TODO Auto-generated method stub
		if (showGraphics){
			showGraphics = false;
		} else {
			showGraphics = true;
		}
		return showGraphics;
	}

}