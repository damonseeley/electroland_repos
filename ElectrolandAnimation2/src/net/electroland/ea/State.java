package net.electroland.ea;

import java.awt.Rectangle;

public class State {

    public Rectangle geometry;
    public double alpha;

    public State(int left, int top, int width, int height, double alpha)
    {
        this.geometry = new Rectangle(left, top, width, height);
        this.alpha = alpha;
    }

    public String toString()
    {
        StringBuffer b = new StringBuffer("State[");
        b.append(geometry).append(',').append(alpha).append(']');
        return b.toString();
    }
}