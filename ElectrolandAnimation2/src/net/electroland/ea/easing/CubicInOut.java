package net.electroland.ea.easing;

import net.electroland.ea.EasingFunction;

public class CubicInOut extends EasingFunction {

    @Override
    public float valueAt(float percentComplete, float start, float finish) {
        percentComplete *=2;
        if (percentComplete < 1){
            return ((finish - start)/2) * percentComplete * percentComplete * percentComplete + start;
        }else{
            percentComplete -= 2;
            return ((finish - start)/2)*(percentComplete*percentComplete*percentComplete + 2) + start;
        }
    }
}