package net.electroland.gotham.processing;

import java.awt.Rectangle;
import java.awt.geom.Point2D;
import java.util.logging.Logger;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.QuinticInOut;
import net.electroland.utils.lighting.canvas.ELUPApplet;

public class East extends ELUPApplet {

    private static final long serialVersionUID = 449793686955037866L;
    static Logger logger = Logger.getLogger("West");

    private Rectangle syncArea;
    private Point2D center;

    private boolean isGrowing = true;
    private long startTime = -1;
    private float smallRadius, largeRadius;

    // the period that the circle expands/contrats
    final static long DURATION_MILLIS = 1000;
    // the easing function for expanding/contracting
    private EasingFunction ef = new QuinticInOut();

    @Override
    public void setup() {
        // syncArea is the area of the screen that will be synced to the lights.
        syncArea = this.getSyncArea();
        // our circle's center will be the middel of the sync area.
        center = new Point2D.Double(syncArea.x + .5 * syncArea.width, 
                                    syncArea.y + .5 * syncArea.height);
        // it will beat between these radii
        smallRadius = .05f * syncArea.height;
        largeRadius = .8f * syncArea.height;
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
        color(255);
        fill(255);
        rect((float)center.getX() - .5f * radius, (float)center.getY() - .5f * radius, radius, radius);
    }
}
