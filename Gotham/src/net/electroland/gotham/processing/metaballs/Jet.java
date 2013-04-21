package net.electroland.gotham.processing.metaballs;

import processing.core.PVector;

public class Jet {

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
                System.out.println("reverse allowed here");
                currentForce *= (Math.random() > .5 ? 1.0f : -1.0f);
            }
            float seconds = (float)(Math.random() * maxDurationSeconds * 1000);
            resetTime    = (long)seconds + System.currentTimeMillis(); 
        }
        return currentForce;
    }

    public String toString(){
        return "Jet[source=" + source + ", angle=" + degrees + ", force=" + baseForce + "]";
    }
}