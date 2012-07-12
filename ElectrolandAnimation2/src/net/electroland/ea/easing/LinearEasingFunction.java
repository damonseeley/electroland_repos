package net.electroland.ea.easing;

public class LinearEasingFunction extends EasingFunction {
    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (percentComplete * (finish - start)) + start;
   }
}