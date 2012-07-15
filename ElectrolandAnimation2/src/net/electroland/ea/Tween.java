package net.electroland.ea;

import net.electroland.ea.easing.Linear;

public class Tween {

    protected Change reqXchange, reqYchange, reqWchange, reqHchange, reqAchange;
    protected EasingFunction FofX, FofY, FofW, FofH, FofA;
//    protected ClipState finalState;

    /**
     * By default, create a linear tweening Change
     */
    public Tween()
    {
        EasingFunction ef = new Linear();
        FofX = ef;
        FofY = ef;
        FofW = ef;
        FofH = ef;
        FofA = ef;
    }

    /**
     * Create a Change that tweens using the specified easing function.
     * @param easingFunction
     */
    public Tween(EasingFunction easingFunction)
    {
        FofX = easingFunction;
        FofY = easingFunction;
        FofW = easingFunction;
        FofH = easingFunction;
        FofA = easingFunction;
    }

    protected ClipState nextFrame(ClipState init, float percentComplete){

        // targetValue gets calculated on every frame. a little inefficient, but
        // remember that this tween is applied to multiple Clips potentially.
        // can't store Clip state here.
        int   x = (int)FofX.valueAt(percentComplete, init.geometry.x,      calculateTargetValue(init.geometry.x,      reqXchange));
        int   y = (int)FofY.valueAt(percentComplete, init.geometry.y,      calculateTargetValue(init.geometry.y,      reqYchange));
        int   w = (int)FofW.valueAt(percentComplete, init.geometry.width,  calculateTargetValue(init.geometry.width,  reqWchange));
        int   h = (int)FofH.valueAt(percentComplete, init.geometry.height, calculateTargetValue(init.geometry.height, reqHchange));
        float a =      FofA.valueAt(percentComplete, init.alpha,           calculateTargetValue(init.alpha,           reqAchange));
        return new ClipState(x,y,w,h,a);
    }

    private float calculateTargetValue(float current, Change change){
        if (change == null)
            return current;
        else 
            return change.applyTo(current);
    }

    // easing functions
    public Tween xUsing(EasingFunction ef){
        FofX = ef;
        return this;
    }
    public Tween yUsing(EasingFunction ef){
        FofY = ef;
        return this;
    }
    public Tween widthUsing(EasingFunction ef){
        FofW = ef;
        return this;
    }
    public Tween heightUsing(EasingFunction ef){
        FofH = ef;
        return this;
    }
    public Tween alphaUsing(EasingFunction ef){
        FofA = ef;
        return this;
    }

    // absolute pixel changes
    public Tween xTo(float x){
        reqXchange = new ChangeTo(x);
        return this;
    }
    public Tween yTo(float y){
        reqYchange = new ChangeTo(y);
        return this;
    }
    public Tween widthTo(float width){
        reqWchange = new ChangeTo(width);
        return this;
    }
    public Tween heightTo(float height){
        reqHchange = new ChangeTo(height);
        return this;
    }
    public Tween alphaTo(float alpha){
        reqAchange = new ChangeTo(alpha);
        return this;
    }

    // relative pixel changes
    public Tween xBy(float dx){
        reqXchange = new ChangeBy(dx);
        return this;
    }
    public Tween yBy(float dy){
        reqYchange = new ChangeBy(dy);
        return this;
    }
    public Tween widthBy(float dWidth){
        reqWchange = new ChangeBy(dWidth);
        return this;
    }
    public Tween heightBy(float dHeight){
        reqHchange = new ChangeBy(dHeight);
        return this;
    }
    public Tween alphaBy(float dAlpha){
        reqAchange = new ChangeBy(dAlpha);
        return this;
    }

    // percent changes (dimensions only: do not correct for centering)
    public Tween scaleWidth(float percent)
    {
        reqWchange = new ScaleBy(percent);
        return this;
    }
    public Tween scaleHeight(float percent)
    {
        reqHchange = new ScaleBy(percent);
        return this;
    }
    public Tween scaleAlpha(float percent)
    {
        reqAchange = new ScaleBy(percent);
        return this;
    }
}