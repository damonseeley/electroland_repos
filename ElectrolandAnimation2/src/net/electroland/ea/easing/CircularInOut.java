package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class CircularInOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete *= 2;
        if (percentComplete < 1){
            return -(finish - start)/2 * ((float)Math.sqrt(1 - percentComplete * percentComplete) - 1) + start;
        }else{
            percentComplete -=2;
            return (finish - start)/2 * ((float)Math.sqrt(1 - percentComplete * percentComplete) + 1) + start;
        }
    }
}