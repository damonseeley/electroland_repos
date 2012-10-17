package net.electroland.utils.lighting.canvas;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Rectangle;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.DetectionModel;

import org.apache.log4j.Logger;

import processing.core.PApplet;
import processing.core.PImage;

abstract public class ELUPApplet extends PApplet {

    static Logger logger = Logger.getLogger(ELUPApplet.class);
    private static final long serialVersionUID = -8484348842116122238L;
    private Dimension dim;
    private ProcessingCanvas canvas;
    private boolean showDetectors = true;
    private boolean showRendering = true;
    private DetectionModel showOnly;
    private float scale = 1.0f;

    abstract public void drawELUContent();

    final public void draw(){

        drawELUContent();

        translate(0,0);
        rectMode(CORNER);

        // sync content to lights
        if (canvas != null){
            PImage image = this.get(0, 0, dim.width, dim.height);
            int[] pixels;
            if (image.width == dim.width){
                if (image.height != dim.height){
                    pixels = new int[dim.width * dim.height];
                    // zero pad (if necessary)
                    System.arraycopy(image.pixels, 0, pixels, 0, image.pixels.length);
                }else{
                    pixels = image.pixels;
                }
                canvas.sync(pixels);
            }else{
                logger.error("Cannot sync when Applet is narrower than " + dim.width + " pixels.");
            }
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
        this.rect(- 1, - 1, dim.width + 2, dim.height + 2);

        // show detectors (if user clicked preference for detetors)
        if (showDetectors && canvas != null){
            for (CanvasDetector cd : canvas.getDetectors()){
                // filter to show only a specific type of Detector (by DetectionModel)
                if (showOnly == null || cd.getDetectorModel().getClass() == showOnly.getClass()){
                    noStroke();
                    if (cd.getLatestState() == (byte)0){
                        stroke(50,50,50);
                        noFill();
                    }else{
                        stroke(50,50,50);
                        noStroke();
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

    public Dimension getSyncArea(){
        return new Dimension(dim);
    }

    public void setDetectorScale(float scale){
        this.scale = scale;
    }

    public void setSyncArea(Dimension dimensions){
        this.dim = dimensions;
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