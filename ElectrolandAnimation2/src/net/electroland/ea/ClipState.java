package net.electroland.ea;

import java.awt.Rectangle;

/**
 * A state of a Clip.  The state is defined as
 * the Clip's geometry (x,y,width,height) as well as it's alpha.
 * @author production
 *
 */
public class ClipState {

    public Rectangle geometry;
    public float alpha;

    public ClipState(int left, int top, int width, int height, float alpha)
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