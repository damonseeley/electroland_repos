package net.electroland.ea.easing;

public class DelayedJumpEasingFunction extends EasingFunction {

    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (int)percentComplete == 1 ? finish : start;
    }

}
