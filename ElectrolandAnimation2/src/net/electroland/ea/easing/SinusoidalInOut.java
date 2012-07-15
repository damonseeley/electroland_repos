package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class SinusoidalInOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return -(finish - start)/2 * ((float)Math.cos(Math.PI * percentComplete) - 1) + start;
    }
}