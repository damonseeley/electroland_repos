package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class Linear extends EasingFunction {
    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return (percentComplete * (finish - start)) + start;
   }
}