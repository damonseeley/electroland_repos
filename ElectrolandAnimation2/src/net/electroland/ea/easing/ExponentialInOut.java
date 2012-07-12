package net.electroland.ea.easing;

public class ExponentialInOut extends EasingFunction {

    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        percentComplete *= 2;
        if (percentComplete < 1){
            return ((finish - start) / 2) * Math.pow(2, 10 * (percentComplete - 1)) + start;
        }else{
            return ((finish - start) / 2) * (-Math.pow(2, -10 * percentComplete) + 2) + start;
        }
    }
}