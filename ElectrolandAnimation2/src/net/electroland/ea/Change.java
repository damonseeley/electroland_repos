package net.electroland.ea;


abstract public class Change {

    protected Double toLeft, toTop, toWidth, toHeight;
    protected Double toAlpha;
    protected Double byLeft, byTop, byWidth, byHeight;
    protected Double byAlpha;
    protected Double scaleWidth, scaleHeight;
    protected Double scaleAlpha;

    abstract public State nextState(State init, double percentComplete);

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
        byWidth = dWidth;
        return this;
    }
    public Change heightBy(double dHeight){
        byHeight = dHeight;
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