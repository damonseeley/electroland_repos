package net.electroland.ea.easing;

public class CubicInOut extends EasingFunction {

    @Override
    public double valueAt(double percentComplete, double start, double finish) {
        percentComplete *=2;
        if (percentComplete < 1){
            return ((finish - start)/2) * percentComplete * percentComplete * percentComplete + start;
        }else{
            percentComplete -= 2;
            return ((finish - start)/2)*(percentComplete*percentComplete*percentComplete + 2) + start;
        }
    }
}