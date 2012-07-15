package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class CircularOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete--;
        return (finish - start) * (float)Math.sqrt(1 - percentComplete * percentComplete) + start;
    }
}