package net.electroland.utils.lighting.canvas;

import java.awt.Rectangle;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.InvalidPixelGrabException;

public class ProcessingCanvas extends ELUCanvas2D {

    private boolean isSyncing = false;
    public static final int NONE  = -0;
    public static final int ALL   = 0;
    public static final int RED   = 1;
    public static final int GREEN = 2;
    public static final int BLUE  = 3;
    private ELUPApplet applet;

    @Override
    public void configure(ParameterMap props) throws OptionException {

        super.configure(props);

        int x = props.getRequiredInt("x");
        int y = props.getRequiredInt("y");
        int fps = props.getRequiredInt("fps");
        Object appletObj = props.getRequiredClass("applet");

        if (appletObj instanceof ELUPApplet){
            applet = (ELUPApplet)appletObj;
            applet.setSyncArea(new Rectangle(x, y, this.getDimensions().width, this.getDimensions().height));
            applet.setSyncCanvas(this);
            applet.init();
            applet.frameRate(fps);
            applet.setVisible(true);
        }
    }

    // NOT ALLOWED!  sync will automatically be called by the ELUPapplet after
    // each frame is rendered.
    final public CanvasDetector[] sync(int[] pixels) {
        throw new RuntimeException("ProcessingCanvases cannot be manually sync'ed.");
    }

    protected CanvasDetector[] pAppletSync(int[] pixels) throws InvalidPixelGrabException {
        if (isSyncing){
            return super.sync(pixels);
        }else{
            return super.getDetectors();
        }
    }

    public void setSyncState(boolean isSyncing){
        this.isSyncing = isSyncing;
    }
    public void setOverlay(int overlayState){
        applet.overlayState = overlayState;
    }
}