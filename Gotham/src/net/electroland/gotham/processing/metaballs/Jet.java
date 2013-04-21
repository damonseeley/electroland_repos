package net.electroland.gotham.processing.metaballs;

import processing.core.PVector;

public class Jet {

    private PVector source;
    private float degrees, baseForce, currentForce, maxDurationSeconds;
    private long resetTime = 0;

    public Jet(PVector source, float degreesFromNorthClockwise, float baseForce, float maxDurationSeconds){
        this.source = source;
        this.degrees = degreesFromNorthClockwise;
        this.baseForce = baseForce;
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
            resetTime    = (int)((Math.random() * maxDurationSeconds) + System.currentTimeMillis()); 
        }
        return currentForce;
    }

    public String toString(){
        return "Jet[source=" + source + ", angle=" + degrees + ", force=" + baseForce + "]";
    }
}