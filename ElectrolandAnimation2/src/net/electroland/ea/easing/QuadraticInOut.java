package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class QuadraticInOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete *= 2;
        if (percentComplete < 1){
            return (finish - start)/2 * percentComplete * percentComplete + start;
        }else{
            percentComplete--;
            return -(finish - start)/2 * (percentComplete * (percentComplete - 2) - 1) + start;
        }
    }
}