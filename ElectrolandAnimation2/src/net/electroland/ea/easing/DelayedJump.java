package net.electroland.ea.easing;

public class DelayedJump extends EasingFunction {

    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (int)percentComplete == 1 ? finish : start;
    }

}
