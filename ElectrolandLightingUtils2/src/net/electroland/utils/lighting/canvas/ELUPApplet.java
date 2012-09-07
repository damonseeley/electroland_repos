package net.electroland.utils.lighting.canvas;

import java.awt.Rectangle;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.InvalidPixelGrabException;

import processing.core.PApplet;

abstract public class ELUPApplet extends PApplet {

    private static final long serialVersionUID = -8484348842116122238L;
    private Rectangle area;
    private ProcessingCanvas canvas;
    private boolean overlayDetectors = true;
    protected int overlayState = ProcessingCanvas.ALL;

    abstract public void drawELUContent();

    final public void draw(){
        drawELUContent();
        if (canvas != null){
            try {
                canvas.pAppletSync(this.get(area.x, area.y, area.width, area.height).pixels);
            } catch (InvalidPixelGrabException e) {
                e.printStackTrace();
            }
        }
        if (overlayDetectors){
            for (CanvasDetector cd : canvas.getDetectors()){

                // TODO: draw detector (R,G,B as selected)
                //  Remember to offset by x,y
                switch(overlayState){
                    case(ProcessingCanvas.ALL):
                        break;
                    case(ProcessingCanvas.RED):
                        break;
                    case(ProcessingCanvas.GREEN):
                        break;
                    case(ProcessingCanvas.BLUE):
                        break;
                }
            }
        }
    }

    public void setSyncCanvas(ProcessingCanvas canvas){
        this.canvas = canvas;
    }

    public Rectangle getSyncArea(){
        return area;
    }

    public void setSyncArea(Rectangle area){
        this.area = area;
    }

    public void setOverlayDetectors(boolean overlayDetectors){
        this.overlayDetectors = overlayDetectors;
    }
}