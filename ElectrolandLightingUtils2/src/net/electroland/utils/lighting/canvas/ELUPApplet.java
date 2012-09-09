package net.electroland.utils.lighting.canvas;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.logging.Logger;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.DetectionModel;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import processing.core.PApplet;

abstract public class ELUPApplet extends PApplet {

    static Logger logger = Logger.getLogger("ELUPApplet");
    private static final long serialVersionUID = -8484348842116122238L;
    private Rectangle area;
    private ProcessingCanvas canvas;
    private boolean overlayDetectors = true;
    protected int overlayState = ProcessingCanvas.ALL;
    private DetectionModel showOnly;

    abstract public void drawELUContent();

    final public void draw(){

        drawELUContent();

        if (canvas != null){
            canvas.sync(this.get(area.x, area.y, area.width, area.height).pixels);
        }
        if (overlayDetectors && canvas != null){
            for (CanvasDetector cd : canvas.getDetectors()){
                
                if (showOnly == null || cd.getDetectorModel().getClass() == showOnly.getClass()){
                    stroke(100,100,100);
                    strokeWeight(1);
                    if (cd.getLatestState() == 0){
                        noFill();
                    }else{
                        Color c = cd.getDetectorModel().getColor(cd.getLatestState());
                        fill(c.getRGB());
                    }
                    Rectangle drect = (Rectangle)cd.getBoundary();
                    rect(drect.x, drect.y, 5, 5);
                }
            }
        }
        stroke(255);
        strokeWeight(1);
        noFill();
        this.rect(area.x - 1, area.y - 1, area.width + 2, area.height + 2);
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

    public void showOnly(DetectionModel detectionModel){
        showOnly = detectionModel;
    }
    public void showAll(){
        showOnly = null;
    }
}