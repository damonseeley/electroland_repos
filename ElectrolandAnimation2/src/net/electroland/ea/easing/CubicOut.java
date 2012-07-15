package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class CubicOut extends EasingFunction {
    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete--;
        return (finish - start)*(percentComplete*percentComplete*percentComplete + 1) + start;
    }
}