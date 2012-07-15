package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class QuadraticIn extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return (finish - start) * percentComplete * percentComplete + start;
    }
}