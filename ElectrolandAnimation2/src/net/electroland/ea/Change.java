package net.electroland.ea;

import net.electroland.ea.easing.EasingFunction;
import net.electroland.ea.easing.Linear;

/**
 * A change represents a set of changes to be applied to a Clip over time.
 * Changes can be specified either absolutely (e.g., xTo(10)) or relatively
 * (e.g., xBy(50)) or scaled (scaleWidth(.5)).  x, y, width, height and
 * alpha can all be manipulated.
 * 
 * Concrete instances of Change define the time-based assignment of changes.
 * 
 * @author production
 *
 */
public class Change {

    protected Double toLeft, toTop, toWidth, toHeight;
    protected Double toAlpha;
    protected Double byLeft, byTop, byWidth, byHeight;
    protected Double byAlpha;
    protected Double scaleWidth, scaleHeight;
    protected Double scaleAlpha;
    protected EasingFunction easingFofX, easingFofY, easingFofW, easingFofH, easingFofA;

    public Change()
    {
        EasingFunction defaultEase = new Linear();
        this.easingFofX = defaultEase;
        this.easingFofY = defaultEase;
        this.easingFofW = defaultEase;
        this.easingFofH = defaultEase;
        this.easingFofA = defaultEase;
    }

    public Change(EasingFunction easingFunction)
    {
        this.easingFofX = easingFunction;
        this.easingFofY = easingFunction;
        this.easingFofW = easingFunction;
        this.easingFofH = easingFunction;
        this.easingFofA = easingFunction;
    }

    public State nextState(State init, double percentComplete){
        int x = (int)easingFofX.valueAt(percentComplete, init.geometry.x, this.getTargetState(init).geometry.x);
        int y = (int)easingFofY.valueAt(percentComplete, init.geometry.y, this.getTargetState(init).geometry.y);
        int w = (int)easingFofW.valueAt(percentComplete, init.geometry.width, this.getTargetState(init).geometry.width);
        int h = (int)easingFofH.valueAt(percentComplete, init.geometry.height, this.getTargetState(init).geometry.height);
        double a = (double)easingFofA.valueAt(percentComplete, init.alpha, this.getTargetState(init).alpha);
        return new State(x,y,w,h,a);
    }

    public State getTargetState(State init)
    {
        double x = target(init.geometry.x, toLeft, byLeft, null);
        double y = target(init.geometry.y, toTop, byTop, null);
        double w = target(init.geometry.width, toWidth, byWidth, scaleWidth);
        double h = target(init.geometry.height, toHeight, byHeight, scaleHeight);
        double a = (float)target(init.alpha, toAlpha, byAlpha, scaleAlpha);

        State next = new State((int)x,(int)y,(int)w,(int)h,a);
        return next;
    }

    private double target(double current, Double absolute, Double relative, Double percent)
    {
        if (absolute != null)
            return absolute;
        else if (relative != null)
            return current + relative;
        else if (percent != null)
            return current * percent;
        else return current;
    }

    // absolute pixel changes
    public Change xTo(double left){
        toLeft = left;
        return this;
    }
    public Change xTo(double left, EasingFunction ef){
        toLeft = left;
        easingFofX = ef;
        return this;
    }
    public Change yTo(double top){
        toTop = top;
        return this;
    }
    public Change yTo(double top, EasingFunction ef){
        toTop = top;
        easingFofY = ef;
        return this;
    }
    public Change widthTo(double width){
        toWidth = width;
        return this;
    }
    public Change widthTo(double width, EasingFunction ef){
        toWidth = width;
        easingFofW = ef;
        return this;
    }
    public Change heightTo(double height){
        toHeight = height;
        return this;
    }
    public Change heightTo(double height, EasingFunction ef){
        toHeight = height;
        easingFofH = ef;
        return this;
    }
    public Change alphaTo(double alpha){
        toAlpha = alpha;
        return this;
    }
    public Change alphaTo(double alpha, EasingFunction ef){
        toAlpha = alpha;
        easingFofA = ef;
        return this;
    }

    // relative pixel changes
    public Change xBy(double dLeft){
        byLeft = dLeft;
        return this;
    }
    public Change xBy(double dLeft, EasingFunction ef){
        byLeft = dLeft;
        easingFofX = ef;
        return this;
    }
    public Change yBy(double dTop){
        byTop = dTop;
        return this;
    }
    public Change yBy(double dTop, EasingFunction ef){
        byTop = dTop;
        easingFofY = ef;
        return this;
    }
    public Change widthBy(double dWidth){
        byWidth = dWidth;
        return this;
    }
    public Change widthBy(double dWidth, EasingFunction ef){
        byWidth = dWidth;
        easingFofW = ef;
        return this;
    }
    public Change heightBy(double dHeight){
        byHeight = dHeight;
        return this;
    }
    public Change heightBy(double dHeight, EasingFunction ef){
        byHeight = dHeight;
        easingFofH = ef;
        return this;
    }
    public Change alphaBy(double dAlpha){
        byAlpha = dAlpha;
        return this;
    }
    public Change alphaBy(double dAlpha, EasingFunction ef){
        byAlpha = dAlpha;
        easingFofA = ef;
        return this;
    }

    // percent changes
    public Change scaleWidth(double percent)
    {
        scaleWidth = percent;
        return this;
    }
    public Change scaleWidth(double percent, EasingFunction ef){
        scaleWidth = percent;
        easingFofW = ef;
        return this;
    }
    public Change scaleHeight(double percent)
    {
        scaleHeight = percent;
        return this;
    }
    public Change scaleHeight(double percent, EasingFunction ef){
        scaleHeight = percent;
        easingFofH  = ef;
        return this;
    }
    public Change scaleAlpha(double percent)
    {
        scaleAlpha = percent;
        return this;
    }
    public Change scaleAlpha(double percent, EasingFunction ef){
        scaleAlpha = percent;
        easingFofA = ef;
        return this;
    }
    public String toString()
    {
        StringBuffer b = new StringBuffer();
        b.append("to[").append(toLeft).append(',').append(toTop).append(',');
        b.append(toWidth).append(',').append(toHeight).append(',').append(toAlpha).append(']');
        b.append(" by[").append(byLeft).append(',').append(byTop).append(',');
        b.append(byWidth).append(',').append(byHeight).append(',').append(byAlpha).append(']');
        b.append(" scale[").append(scaleWidth).append(',').append(scaleHeight).append(',').append(scaleAlpha).append(']');
        return b.toString();
    }
}