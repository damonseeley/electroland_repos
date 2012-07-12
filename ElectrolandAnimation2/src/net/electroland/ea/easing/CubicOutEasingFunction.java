package net.electroland.ea.easing;

public class CubicOutEasingFunction extends EasingFunction {
    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        percentComplete--;
        return (finish - start)*(percentComplete*percentComplete*percentComplete + 1) + start;
    }
}