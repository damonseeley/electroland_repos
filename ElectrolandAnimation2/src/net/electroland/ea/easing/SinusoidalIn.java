package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class SinusoidalIn extends EasingFunction{

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return -(finish - start) * (float)Math.cos(percentComplete * (Math.PI/2)) + finish;
    }
}