package net.electroland.utils.lighting.canvas;

import java.awt.BorderLayout;
import java.awt.Rectangle;

import javax.swing.JFrame;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class ProcessingCanvas extends ELUCanvas2D {

    public static final int NONE  = -0;
    public static final int ALL   = 0;
    public static final int RED   = 1;
    public static final int GREEN = 2;
    public static final int BLUE  = 3;
    private ELUPApplet applet;

    @Override
    public void configure(ParameterMap props) throws OptionException {

        super.configure(props);

        int fps = props.getRequiredInt("fps");
        Object appletObj = props.getRequiredClass("applet");

        System.out.println("instantiating applet " + appletObj);
        if (appletObj instanceof ELUPApplet){
            System.out.println("instantiating applet " + appletObj);
            applet = (ELUPApplet)appletObj;

            // make sure this is precedes init(). otherwise there is a race
            // condition against setup().
            applet.setSyncArea(new Rectangle(0, 0, this.getDimensions().width, this.getDimensions().height));

            JFrame f = new JFrame();
            f.setTitle(props.get("applet"));
            f.setLayout(new BorderLayout());
            f.setSize(this.getDimensions());
            f.add(applet, BorderLayout.CENTER);
            applet.init();
            applet.frameRate(fps);
            applet.setSyncCanvas(this);
            f.setVisible(true);
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            System.out.println("and we're done here");
        }
    }

    public void setOverlay(int overlayState){
        applet.overlayState = overlayState;
    }

    public ELUPApplet getApplet(){
        return applet;
    }
}