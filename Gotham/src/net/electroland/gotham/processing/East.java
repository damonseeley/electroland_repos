package net.electroland.gotham.processing;

import java.awt.Color;
import java.awt.Rectangle;
import java.awt.geom.Point2D;
import java.util.logging.Logger;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.QuinticInOut;
import net.electroland.ea.easing.SinusoidalInOut;
import net.electroland.utils.lighting.canvas.ELUPApplet;

public class East extends ELUPApplet {

    private static final long serialVersionUID = 449793686955037866L;
    static Logger logger = Logger.getLogger("West");

    private Rectangle syncArea;
    private Point2D center;

    private boolean isGrowing = true;
    private long startTime = -1;
    private float smallRadius, largeRadius;
    private Color nextColor, currentColor = Color.white;

    // the period that the circle expands/contracts
    final static long DURATION_MILLIS = 3000;
    // the easing function for expanding/contracting
    private EasingFunction ef = new QuinticInOut();
    private EasingFunction cf = new SinusoidalInOut();

    @Override
    public void setup() {
        // syncArea is the area of the screen that will be synced to the lights.
        syncArea = this.getSyncArea();
        // our square's center will be the middle of the sync area.
        center = new Point2D.Double(syncArea.x + .5 * syncArea.width, 
                                    syncArea.y + .5 * syncArea.height);
        // it will beat between these radii
        smallRadius = .05f * syncArea.height;
        largeRadius = 1.2f * syncArea.height;
        nextColor = randomColor();
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
            currentColor = nextColor;
            nextColor = randomColor();
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
        color(cf.valueAt(percentComplete, currentColor.getRed(), nextColor.getRed()), 
              cf.valueAt(percentComplete, currentColor.getGreen(), nextColor.getGreen()), 
              cf.valueAt(percentComplete, currentColor.getBlue(), nextColor.getBlue()));

        fill(cf.valueAt(percentComplete, currentColor.getRed(), nextColor.getRed()), 
                cf.valueAt(percentComplete, currentColor.getGreen(), nextColor.getGreen()), 
                cf.valueAt(percentComplete, currentColor.getBlue(), nextColor.getBlue()));

        rect((float)center.getX() - .5f * radius, (float)center.getY() - .5f * radius, radius, radius);
    }

    public Color randomColor() {
        return new Color((int)(Math.random() * 255), (int)(Math.random() * 255), (int)(Math.random() * 255));
    }

}
