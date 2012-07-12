package net.electroland.ea;

import net.electroland.ea.easing.EasingFunction;
import net.electroland.ea.easing.LinearEasingFunction;

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
    protected EasingFunction easingFunction;

    public Change()
    {
        this.easingFunction = new LinearEasingFunction();
    }

    public Change(EasingFunction easingFunction)
    {
        this.easingFunction = easingFunction;
    }

    public State nextState(State init, double percentComplete){
        int x = (int)easingFunction.valueAt(percentComplete, init.geometry.x, this.getTargetState(init).geometry.x);
        int y = (int)easingFunction.valueAt(percentComplete, init.geometry.y, this.getTargetState(init).geometry.y);
        int w = (int)easingFunction.valueAt(percentComplete, init.geometry.width, this.getTargetState(init).geometry.width);
        int h = (int)easingFunction.valueAt(percentComplete, init.geometry.height, this.getTargetState(init).geometry.height);
        double a = (double)easingFunction.valueAt(percentComplete, init.alpha, this.getTargetState(init).alpha);
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
    public Change yTo(double top){
        toTop = top;
        return this;
    }
    public Change widthTo(double width){
        toWidth = width;
        return this;
    }
    public Change heightTo(double height){
        toHeight = height;
        return this;
    }
    public Change alphaTo(double alpha){
        toAlpha = alpha;
        return this;
    }

    // relative pixel changes
    public Change xBy(double dLeft){
        byLeft = dLeft;
        return this;
    }
    public Change yBy(double dTop){
        byTop = dTop;
        return this;
    }
    public Change widthBy(double dWidth){
        byWidth = dWidth; // erm.  this seems more intuitive as byWidth *= dWidth;
        return this;
    }
    public Change heightBy(double dHeight){
        byHeight = dHeight; // erm.  this seems more intuitive as byHeight *= dHeight;
        return this;
    }
    public Change alphaBy(double dAlpha){
        byAlpha = dAlpha;
        return this;
    }

    // percent changes
    public Change scaleWidth(double percent)
    {
        scaleWidth = percent;
        return this;
    }
    public Change scaleHeight(double percent)
    {
        scaleHeight = percent;
        return this;
    }
    public Change scaleAlpha(double percent)
    {
        scaleAlpha = percent;
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