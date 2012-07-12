package net.electroland.ea.easing;

public class Linear extends EasingFunction {
    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        return (percentComplete * (finish - start)) + start;
   }
}