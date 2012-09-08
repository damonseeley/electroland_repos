package net.electroland.gotham.processing;

import java.awt.Rectangle;
import java.awt.geom.Point2D;
import java.util.logging.Logger;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.SinusoidalInOut;
import net.electroland.utils.lighting.canvas.ELUPApplet;

public class West extends ELUPApplet {

    private static final long serialVersionUID = 449793686955037866L;
    static Logger logger = Logger.getLogger("West");

    private Point2D center;
    private Rectangle syncArea;
    private long startChange = -1;
    private float smallRadius, largeRadius;
    private boolean isGrowing = true;
    private EasingFunction ef = new SinusoidalInOut();
    final static long DURATION_MILLIS = 1000;

    @Override
    public void setup() {
        syncArea = this.getSyncArea();
        center = new Point2D.Double(syncArea.x + .5 * syncArea.width, 
                                    syncArea.y + .5 * syncArea.height);
        smallRadius = .05f * syncArea.height;
        largeRadius = .4f * syncArea.height;
    }

    @Override
    public void drawELUContent() {

        // erase background
        color(255);
        fill(255);
        rect(0,0,this.getWidth(), this.getHeight());

        // check to see if the growth cycle should switch polarity
        if (System.currentTimeMillis() - startChange > DURATION_MILLIS){
            isGrowing = !isGrowing;
            startChange = System.currentTimeMillis();
        }

        // calculate the current radius based on time.
        float radius;
        float percentComplete = (System.currentTimeMillis() - startChange) / (float)DURATION_MILLIS;
        if (isGrowing){
            radius = ef.valueAt(percentComplete, smallRadius, largeRadius);
        }else{
            radius = ef.valueAt(percentComplete, largeRadius, smallRadius);
        }

        // paint a circle
        color(0);
        fill(0);
        ellipse((float)center.getX(), (float)center.getY(), radius, radius);
    }
}
