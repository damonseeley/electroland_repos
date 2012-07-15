package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class ExponentialInOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete *= 2;
        if (percentComplete < 1){
            return ((finish - start) / 2) * (float)Math.pow(2, 10 * (percentComplete - 1)) + start;
        }else{
            return ((finish - start) / 2) * (-(float)Math.pow(2, -10 * percentComplete) + 2) + start;
        }
    }
}