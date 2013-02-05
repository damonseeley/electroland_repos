package net.electroland.gotham.processing.metaballs;

import net.electroland.ea.EasingFunction;
import processing.core.PVector;

public class Wind {

    private float degreesFromNorthClockwise, finalDegrees;
    private float maxStrength, holdDurationSec, inDurationSec, outDurationSec;
    private EasingFunction blowingIn, blowingOut;
    private long startTime;

    public Wind(float degreesFromNorthClockwise, float finalDegrees,
                float maxStrength, float holdDurationSec, 
                float inDurationSec, EasingFunction blowingIn, 
                float outDurationSec, EasingFunction blowingOut){

        this.finalDegrees               = finalDegrees;
        this.degreesFromNorthClockwise  = degreesFromNorthClockwise;
        this.inDurationSec              = inDurationSec;
        this.outDurationSec             = outDurationSec;
        this.holdDurationSec            = holdDurationSec;
        this.maxStrength                = maxStrength;
        this.blowingIn                  = blowingIn;
        this.blowingOut                 = blowingOut;
    }

    public PVector getSource(){

        float halfLife = (inDurationSec + holdDurationSec + outDurationSec) / 2.0f;
        float angle;
        if (age() < halfLife){
            float percentComplete = (age() / halfLife ) * .5f;
            angle = blowingIn.valueAt(percentComplete, degreesFromNorthClockwise, finalDegrees);
        }else{
            float percentComplete = (((age() - halfLife) / halfLife) * .5f) + .5f;
            angle = blowingOut.valueAt(percentComplete > 1 ? 1.0f : percentComplete, degreesFromNorthClockwise, finalDegrees);
        }
        return PVector.fromAngle((float)((angle - 90) * (Math.PI/180)));
    }

    public float getStrength(){

        if (age() <= inDurationSec){
            float percentComplete = age() / inDurationSec;
            return blowingIn.valueAt(percentComplete > 1 ? 1.0f : percentComplete, 0, maxStrength);
        } else if (age() <= (inDurationSec + holdDurationSec)){
            return maxStrength;
        } else {
            float percentComplete = (age() - inDurationSec - holdDurationSec) / outDurationSec;
            return blowingOut.valueAt(percentComplete > 1 ? 1.0f : percentComplete, maxStrength, 0);
        }  
    }

    public Wind reset(){
        startTime = System.currentTimeMillis();
        return this;
    }

    public boolean isExpired(){
        return age() > (inDurationSec + holdDurationSec + outDurationSec);
    }

    public float age(){
        return ((System.currentTimeMillis() - startTime) / 1000.0f);
    }

    public String toString(){
        return "Wind[degreesFromNorthClockwise=" + this.degreesFromNorthClockwise + ", maxStrength=" + maxStrength + "]";
    }
}