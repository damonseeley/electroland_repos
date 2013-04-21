package net.electroland.gotham.processing.metaballs;

import org.apache.log4j.Logger;

import processing.core.PVector;

public class Jet {

    public static Logger logger = Logger.getLogger(Jet.class);
    private PVector source;
    private float degrees, baseForce, currentForce, maxDurationSeconds;
    private long resetTime = 0;
    private boolean allowReverse = true;

    public Jet(PVector source, float degreesFromNorthClockwise, float baseForce, float maxDurationSeconds, boolean allowReverse){
        this.source       = source;
        this.degrees      = degreesFromNorthClockwise;
        this.baseForce    = baseForce;
        this.maxDurationSeconds = maxDurationSeconds;
        this.allowReverse = allowReverse;
    }

    public PVector getOrigin(){
        return source;
    }

    public PVector getForceVector(){
        return PVector.fromAngle((float)((degrees - 90) * (Math.PI/180)));
    }

    public float getStrength(){
        if (System.currentTimeMillis() > resetTime){
            currentForce = (float)(Math.random() * baseForce);
            if (allowReverse){
                currentForce *= (Math.random() > .5 ? 1.0f : -1.0f);
            }
            float seconds = (float)(Math.random() * maxDurationSeconds * 1000);
            resetTime    = (long)seconds + System.currentTimeMillis(); 
            logger.debug("new force " + currentForce + " for " + (resetTime - System.currentTimeMillis()) + " millis.");
        }
        return currentForce;
    }

    public String toString(){
        return "Jet[source=" + source + ", angle=" + degrees + ", force=" + baseForce + "]";
    }
}