package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class ExponentialIn extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return (finish - start) * (float)Math.pow(2, 10 * (percentComplete - 1)) + start;
    }
}