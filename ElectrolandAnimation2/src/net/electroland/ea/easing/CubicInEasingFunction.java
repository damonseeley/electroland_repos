package net.electroland.ea.easing;

public class CubicInEasingFunction extends EasingFunction {
    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (finish - start) * percentComplete * percentComplete * percentComplete + start;
    }
}