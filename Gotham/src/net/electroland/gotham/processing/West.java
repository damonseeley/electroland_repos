package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.awt.geom.Point2D;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.SinusoidalInOut;

import org.apache.log4j.Logger;

public class West extends GothamPApplet {

    private static final long serialVersionUID = 449793686955037866L;
    static Logger logger = Logger.getLogger(GothamPApplet.class);

    private Dimension syncArea;
    private Point2D center;

    private boolean isGrowing = true;
    private long startTime = -1;
    private float smallRadius, largeRadius;

    // the period that the circle expands/contracts
    final static long DURATION_MILLIS = 2000;
    // the easing function for expanding/contracting
    private EasingFunction ef = new SinusoidalInOut();

    @Override
    public void setup() {
        // syncArea is the area of the screen that will be synced to the lights.
        syncArea = this.getSyncArea();
        // our circle's center will be the middle of the sync area.
        center = new Point2D.Double(.5 * syncArea.width, 
                                    .5 * syncArea.height);
        // it will beat between these radii
        smallRadius = .05f * syncArea.height;
        largeRadius = 1.2f * syncArea.height;
    }

    @Override
    public void drawELUContent() {

        // erase background
        color(0);
        fill(0);
        rect(0,0,this.getWidth(), this.getHeight());

        // check to see if the growth cycle should switch polarity
        if (System.currentTimeMillis() - startTime > DURATION_MILLIS){
            isGrowing = !isGrowing;
            startTime = System.currentTimeMillis();
        }

        // calculate the current radius based on time.
        float radius;
        float percentComplete = (System.currentTimeMillis() - startTime) / (float)DURATION_MILLIS;
        if (isGrowing){
            radius = ef.valueAt(percentComplete, smallRadius, largeRadius);
        }else{
            radius = ef.valueAt(percentComplete, largeRadius, smallRadius);
        }

        // paint a circle
        color(50, 255, 150);
        fill(50, 255, 150);
        ellipse((float)center.getX(), (float)center.getY(), radius, radius);
    }
}