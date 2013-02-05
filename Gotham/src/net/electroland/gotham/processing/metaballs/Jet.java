package net.electroland.gotham.processing.metaballs;

import processing.core.PVector;

public class Jet {

    private PVector source;
    private float degrees, baseForce;

    public Jet(PVector source, float degreesFromNorthClockwise, float baseForce){
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
        return baseForce;
    }

    public String toString(){
        return "Jet[source=" + source + ", angle=" + degrees + ", force=" + baseForce + "]";
    }
}