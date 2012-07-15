package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class DelayedJump extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        return (int)percentComplete == 1 ? finish : start;
    }

}
