package net.electroland.gotham.processing.metaballs;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import processing.core.PVector;

public class MetaballGroup {

    protected int id;
    protected List <Metaball>balls;

    private Color baseColor;
    private float hueVariance;
    private int opacity;
    private float scale;

    public MetaballGroup(int id, float hueVariance){
        this.id = id;
        this.hueVariance = hueVariance;
        balls = new ArrayList<Metaball>();
    }

    public void add(Metaball ball){
        balls.add(ball);
        ball.setGroup(this);
        ball.setHueVariance((float)(Math.random() * hueVariance) - (.5f * hueVariance));
    }

    public void advanceIndividualBalls(){
        for (Metaball ball : balls){
            ball.getCenter().add(ball.getVelocity());
        }
    }

    public void applyFriction(float friction){
        for (Metaball ball : balls){
            ball.getVelocity().mult(friction);
        }
    }

    public void applyPointForce(PVector source, float force){
        for (Metaball ball : balls) {
            ball.repell(source, -force, 2);
        }
    }

    public void applyGroupCohesion(float cohesiveness){
        for (Metaball ball : balls){
            ball.repell(centroid(), -cohesiveness, 1);
        }
    }

    public PVector centroid(){
        PVector centroid = new PVector(0, 0, 0);
        for (Metaball ball : balls){
            centroid.add(ball.getCenter());
        }
        centroid.div(balls.size());
        return centroid;
    }

    public void applyPresenceImpactForces(List<PVector> presences, float magnitude){
        for (PVector point : presences){
            for (Metaball ball : balls){
                ball.repell(point, magnitude, 2);
            }
        }
    }

    /**
     * This might be improved by being group to group, rather than ball to ball.
     * 
     * @param allGroups
     * @param magnitude
     */
    public void applyBallToBallRepellForces(List<MetaballGroup> allGroups, float magnitude){
        for (MetaballGroup group : allGroups){//            if (group != this){
                for (Metaball other : group.balls){
                    for (Metaball ball : balls){
                        if (other != ball){
                            ball.repell(other.getCenter(), magnitude, 2);
                        }
                    }
                }
        }
    }

    public void applyGroupToGroupRepellForces(List<MetaballGroup> allGroups, float magnitude){
        for (MetaballGroup group : allGroups){
            if (group != this){
                for (Metaball other : group.balls){
                    PVector force = this.centroid();
                    force.sub(other.getCenter());
                    force.normalize();
                    other.repell(force, magnitude, 2);
                }
            }
        }
    }

    public void applyJetForces(List<Jet> jets, float forceScale){
        for (Metaball ball : balls){
            for (Jet jet : jets){
                PVector origin = jet.getOrigin();
                PVector forceVector   = jet.getForceVector();
                forceVector.add(ball.getCenter());
                ball.repell(origin, forceVector, jet.getStrength() * forceScale, 2);
            }
        }
    }

    public void applyWindForces(Wind wind, float forceScale){
        for (Metaball ball : balls){
            PVector source = wind.getSource();
            source.add(ball.getCenter());
            ball.repell(source, wind.getStrength() * forceScale, 0);
        }
    }

    public void constrainVelocity(float minVelocity, float maxVelocity){
        for (Metaball ball : balls){
            ball.getVelocity().limit(maxVelocity);
            if (ball.getVelocity().mag() < minVelocity) {
                ball.getVelocity().normalize();
                ball.getVelocity().mult(minVelocity);
            }
        }
    }

    public void keepBallsWithinBoundary(Rectangle boundary){
        for (Metaball ball : balls){
            ball.keepWithinBoundaryOf(boundary);
        }
    }

    public float getScale() {
        return scale;
    }
    public void setScale(float scale) {
        this.scale = scale;
    }
    public int getOpacity() {
        return opacity;
    }
    public void setOpacity(int opacity) {
        this.opacity = opacity;
    }
    public void setBaseColor(Color baseColor){
        this.baseColor = baseColor;
    }
    public Color getBaseColor(){
        return baseColor;
    }
}