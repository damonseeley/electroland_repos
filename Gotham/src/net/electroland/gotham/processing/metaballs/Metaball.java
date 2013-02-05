package net.electroland.gotham.processing.metaballs;

import java.awt.Color;
import java.awt.Rectangle;

import processing.core.PVector;

public class Metaball {

    private PVector center;
    private float radius;
    private PVector velocity;
    private float dHue;
    private MetaballGroup group;

    public Metaball(float radius){
        this.radius = radius;
    }

    /**
     * If this metaball fallsoutside of a boundary, reverse it's trajectory.
     * Essentially, a dumb "bounce".
     * 
     * @param range
     */
    public void keepWithinBoundaryOf(Rectangle range){
        // horizontal
        if (left() < range.x){
            setLeft(range.x);
            velocity.x = -velocity.x;
        } else if (right() > range.width + range.x){
            setRight(range.width + range.x);
            velocity.x = -velocity.x;
        }
        // vertical
        if (top() < range.y){
            setTop(range.y);
            velocity.y = -velocity.y;
        } else if (bottom() > range.height + range.y){
            setBottom(range.height + range.y);
            velocity.y = -velocity.y;
        }
    }

    /**
     * repell applies a force using a normalized version of a vector going 
     * through one point and through the center of this ball. repell forces
     * degrade by square of the distance.
     * 
     * @param point
     * @param force
     * @param distancePower - 2 = force degrades by distance square, 3 = distance cubed, etc.
     */
    public void repell(PVector point, float force, int distancePower){

        float distance = PVector.dist(center, point);
        PVector adjustedForce = PVector.sub(center, point);
        adjustedForce.normalize();
        if (distance > 0){ // check for div-by-zero
            adjustedForce.mult(force / (float)(Math.pow(distance, distancePower)));
        } else {
            adjustedForce.mult(force);
        }
        velocity.add(adjustedForce);
    }

    /**
     * In this case, the forceVector is not derived from the source.  E.g.,
     * It's a point source, but it's force is unidirectional. Like a jet.
     * 
     * @param origin
     * @param forceVector
     * @param force
     * @param distancePower - 2 = force degrades by distance square, 3 = distance cubed, etc.
     */
    public void repell(PVector origin, PVector forceVector, float force, int distancePower){
        float distance = PVector.dist(center, origin);
        PVector adjustedForce = PVector.sub(center, forceVector);
        adjustedForce.normalize();
        if (distance > 0){ // check for div-by-zero
            adjustedForce.mult(-force / (float)(Math.pow(distance, distancePower)));
        } else {
            adjustedForce.mult(-force);
        }
        velocity.add(adjustedForce);
    }

    /**
     * Detect physical overalap between one ball and another.
     * 
     * @param ball
     * @return
     */
    public boolean overlaps(Metaball ball){
        float distance = ball.center.dist(this.center);
        return distance < ball.radius || distance < this.radius;
    }

    public Color getColor(){
        float hsb[] = new float[3];
        Color baseColor = group.getBaseColor();
        Color.RGBtoHSB(baseColor.getRed(), 
                        baseColor.getGreen(), 
                        baseColor.getBlue(), 
                        hsb);
        hsb[0] += dHue;
        if (hsb[0] > 1.0) {
            hsb[0] = 1.0f;
        }
        return new Color(Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]));
    }

    public void setHueVariance(float dHue){
        this.dHue = dHue;
    }
    
    public float width(){
        return radius;
    }
    public float height(){
        return radius;
    }

    public MetaballGroup getGroup() {
        return group;
    }
    public void setGroup(MetaballGroup group) {
        this.group = group;
    }

    public PVector getVelocity() {
        return velocity;
    }
    public void setVelocity(PVector velocity) {
        this.velocity = velocity;
    }

    public PVector getCenter(){
        return center;
    }
    public void setCenter(PVector center){
        this.center = center;
    }
    public float left(){
        return center.x - width() / 2;
    }
    public float right(){
        return center.x + width() / 2;
    }
    public float top(){
        return center.y - height() / 2;
    }
    public float bottom(){
        return center.y + height() / 2;
    }

    public void setLeft(float left){
        center.x = width() / 2 + left; 
    }
    public void setRight(float right){
        center.x = right - width() / 2;
    }
    public void setTop(float top){
        center.y = top + height() / 2;
    }
    public void setBottom(float bottom){
        center.y = bottom - height() / 2;
    }

    public String toString(){
        StringBuffer br = new StringBuffer("Metaball[");
        br.append("group=").append(group.id).append(", ");
        br.append("radius=").append(radius).append(", ");
        br.append("position=").append(center).append(", ");
        br.append("velocity=").append(velocity).append(", ");
        br.append(']');
        return br.toString();
    }

}