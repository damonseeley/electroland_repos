package net.electroland.ea.easing;

abstract public class EasingFunction {
    abstract public double valueAt(double percentComplete, double start, double finish);
}
