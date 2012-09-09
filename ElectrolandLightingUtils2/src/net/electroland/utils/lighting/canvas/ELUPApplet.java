package net.electroland.utils.lighting.canvas;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.logging.Logger;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.DetectionModel;
import processing.core.PApplet;

abstract public class ELUPApplet extends PApplet {

    static Logger logger = Logger.getLogger("ELUPApplet");
    private static final long serialVersionUID = -8484348842116122238L;
    private Rectangle area;
    private ProcessingCanvas canvas;
    private boolean showDetectors = true;
    private boolean showRendering = true;
    protected int overlayState = ProcessingCanvas.ALL;
    private DetectionModel showOnly;

    abstract public void drawELUContent();

    final public void draw(){

        drawELUContent();

        // sync content to lights
        if (canvas != null){
            canvas.sync(this.get(area.x, area.y, area.width, area.height).pixels);
        }

        // draw outline of sync area.  
        stroke(255);
        strokeWeight(1);
        // erase the rendering if it was requested not to show it on the UI
        if (showRendering){
            noFill();
        }else{
            fill(0);
        }
        this.rect(area.x - 1, area.y - 1, area.width + 2, area.height + 2);

        // show detectors (if requested)
        if (showDetectors && canvas != null){
            for (CanvasDetector cd : canvas.getDetectors()){
                
                if (showOnly == null || cd.getDetectorModel().getClass() == showOnly.getClass()){
                    strokeWeight(1);
                    if (cd.getLatestState() == 0){
                        stroke(100,100,100);
                        fill(100,100,100);
                    }else{
                        Color c = cd.getDetectorModel().getColor(cd.getLatestState());
                        stroke(c.getRGB());
                        fill(c.getRGB());
                    }
                    // TODO: there's no guarantee that cd.getBoundary is a Rectangle.  It's a Shape object-
                    //       however Processing can't draw java.aw.Shapes.
                    Rectangle drect = (Rectangle)cd.getBoundary();
                    rect(drect.x, drect.y, drect.width, drect.height);
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

    public void setShowDetectors(boolean showDetectors){
        this.showDetectors = showDetectors;
    }

    public void setShowRendering(boolean showRendering){
        this.showRendering = showRendering;
    }

    public void showOnly(DetectionModel detectionModel){
        showOnly = detectionModel;
    }
    public void showAll(){
        showOnly = null;
    }
}