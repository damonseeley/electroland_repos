package net.electroland.ea.easing;

public class ExponentialOut extends EasingFunction {

    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (finish - start) * (-Math.pow(2, -10 * percentComplete) + 1) + start;
    }
}