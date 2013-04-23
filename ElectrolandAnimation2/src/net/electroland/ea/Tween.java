package net.electroland.ea;

import java.awt.Color;

import net.electroland.ea.easing.Linear;

public class Tween {

    protected Change reqXchange, reqYchange, reqWchange, reqHchange, reqAchange, reqHueChange, reqSatChange, reqBrightChange;
    protected EasingFunction FofX, FofY, FofW, FofH, FofA, FofHue, FofSat, FofBright;
    protected int durationMillis;

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
        FofHue = ef;
        FofSat = ef;
        FofBright = ef;
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
        FofHue = easingFunction;
        FofSat = easingFunction;
        FofBright = easingFunction;
    }

    protected ClipState nextFrame(ClipState init, float percentComplete){

        // targetValue gets calculated on every frame. a little inefficient, but
        // remember that this tween is applied to multiple Clips potentially.
        // can't store Clip state here.
        int   x  = (int)FofX.valueAt(percentComplete, init.geometry.x,      calculateTargetValue(init.geometry.x,      reqXchange));
        int   y  = (int)FofY.valueAt(percentComplete, init.geometry.y,      calculateTargetValue(init.geometry.y,      reqYchange));
        int   w  = (int)FofW.valueAt(percentComplete, init.geometry.width,  calculateTargetValue(init.geometry.width,  reqWchange));
        int   h  = (int)FofH.valueAt(percentComplete, init.geometry.height, calculateTargetValue(init.geometry.height, reqHchange));
        float a  =      FofA.valueAt(percentComplete, init.alpha,           calculateTargetValue(init.alpha,           reqAchange));

        Color bg = null;
        if (init.bgcolor != null){
            float hsbVals[] = Color.RGBtoHSB(init.bgcolor.getRed(),
                    init.bgcolor.getGreen(),
                    init.bgcolor.getBlue(), null);

            float hue =     FofHue.valueAt(percentComplete, hsbVals[0],         calculateTargetValue(hsbVals[0],           reqHueChange));
            float sat =     FofHue.valueAt(percentComplete, hsbVals[1],         calculateTargetValue(hsbVals[1],           reqSatChange));
            float bright =  FofHue.valueAt(percentComplete, hsbVals[2],         calculateTargetValue(hsbVals[2],           reqBrightChange));
            bg =      Color.getHSBColor(hue, sat, bright);
        }
        return new ClipState(x,y,w,h,a,bg);
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
    public Tween hueUsing(EasingFunction ef){
        FofHue = ef;
        return this;
    }
    public Tween saturationUsing(EasingFunction ef){
        FofSat = ef;
        return this;
    }
    public Tween brightnessUsing(EasingFunction ef){
        FofBright = ef;
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
    public Tween hueTo(float h){
        reqHueChange = new ChangeTo(h);
        return this;
    }
    public Tween saturationTo(float s){
        reqSatChange = new ChangeTo(s);
        return this;
    }
    public Tween brightnessTo(float b){
        reqBrightChange = new ChangeTo(b);
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
    public Tween hueBy(float h){
        reqHueChange = new ChangeBy(h);
        return this;
    }
    public Tween saturationBy(float s){
        reqSatChange = new ChangeBy(s);
        return this;
    }
    public Tween brightnessBy(float b){
        reqBrightChange = new ChangeBy(b);
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
    public Tween scaleHue(float percent){
        reqHueChange = new ScaleBy(percent);
        return this;
    }
    public Tween scaleSaturation(float percent){
        reqSatChange = new ScaleBy(percent);
        return this;
    }
    public Tween scaleBrightness(float percent){
        reqBrightChange = new ScaleBy(percent);
        return this;
    }
}