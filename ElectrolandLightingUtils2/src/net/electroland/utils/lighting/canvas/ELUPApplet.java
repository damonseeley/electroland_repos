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
    private DetectionModel showOnly;
    private float scale = 1.0f;

    abstract public void drawELUContent();

    final public void draw(){

        drawELUContent();

        // sync content to lights
        if (canvas != null){
            canvas.sync(this.get(area.x, area.y, area.width, area.height).pixels);
        }

        // draw outline of sync area and...
        stroke(255);
        strokeWeight(1);
        // erase the rendering if the user unclicked the preference to show the canvas
        if (showRendering){
            noFill();
        }else{
            fill(0);
        }
        this.rect(area.x - 1, area.y - 1, area.width + 2, area.height + 2);

        // show detectors (if user clicked preference for detetors)
        if (showDetectors && canvas != null){
            for (CanvasDetector cd : canvas.getDetectors()){
                // filter to show only a specific type of Detector (by DetectionModel)
                if (showOnly == null || cd.getDetectorModel().getClass() == showOnly.getClass()){
                    noStroke();
                    if (cd.getLatestState() == (byte)0){
                        fill(100,100,100);
                    }else{
                        Color c = cd.getDetectorModel().getColor(cd.getLatestState());
                        fill(c.getRGB());
                    }
                    // TODO: there's no guarantee that cd.getBoundary is a Rectangle.  It's a Shape object-
                    //       however Processing can't draw java.aw.Shapes.
                    Rectangle drect = (Rectangle)cd.getBoundary();
                    // TODO: this scaling is inefficient.  should probably cache in the canvas detector.
                    float scaledW = drect.width * scale;
                    float scaledH = drect.height * scale;
                    rect(drect.x - (scaledW * .5f), drect.y - (scaledH * .5f), scaledW, scaledH);
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

    public void setDetectorScale(float scale){
        this.scale = scale;
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