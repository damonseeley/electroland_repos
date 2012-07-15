package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class CircularIn extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return -(finish - start) * (float)(Math.sqrt(1f- percentComplete * percentComplete) - 1f) + start;
    }
}