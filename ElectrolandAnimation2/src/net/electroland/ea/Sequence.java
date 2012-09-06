package net.electroland.ea;

import java.util.Vector;

public class Sequence {

    protected Vector<Tween> sequence;
    private Tween current;

    public Sequence(){
        sequence = new Vector<Tween>();
        newState();
    }

    // add a new target state to Tween to
    public Sequence newState(){
        current = new Tween();
        current.durationMillis = 0;
        sequence.add(current);
        return this;
    }
    public int length(){
        return sequence.size();
    }
    public Tween modify(int step){
        return sequence.get(step);
    }
    public void remove(int step){
        sequence.remove(step);
    }

    // some future nice to haves:
    // fork the sequence
    public Sequence duplicate(){
        // TODO: Implement
        throw new RuntimeException("followWith has not been implemented.");
    }
    //  for looping logic
    public Sequence followWith(Sequence next){
        // TODO: Implement
        throw new RuntimeException("followWith has not been implemented.");
    }
    // for synchronization
    public Sequence waitFor(Object announcment){
        // TODO: Implement
        throw new RuntimeException("waitFor has not been implemented.");
    }
    public Sequence announce(Object announcment){
        // TODO: Implement
        throw new RuntimeException("announce has not been implemented.");
    }

    // timing
    public Sequence pause(int millis){
        Tween t = new Tween();
        t.durationMillis = millis;
        sequence.add(t);
        return this;
    }
    public Sequence duration(int millis){
        current.durationMillis = millis;
        return this;
    }

    // easing functions
    public Sequence using(EasingFunction easing){
        current.FofX = easing;
        current.FofY = easing;
        current.FofW = easing;
        current.FofH = easing;
        current.FofA = easing;
        return this;
    }
    public Sequence xUsing(EasingFunction easing){
        current.FofX = easing;
        return this;
    }
    public Sequence yUsing(EasingFunction easing){
        current.FofY = easing;
        return this;
    }
    public Sequence widthUsing(EasingFunction easing){
        current.FofW = easing;
        return this;
    }
    public Sequence heightUsing(EasingFunction easing){
        current.FofH = easing;
        return this;
    }
    public Sequence alphaUsing(EasingFunction easing){
        current.FofA = easing;
        return this;
    }

    // absolute changes
    public Sequence xTo(float value){
        current.xTo(value);
        return this;
    }
    public Sequence yTo(float value){
        current.yTo(value);
        return this;
    }
    public Sequence widthTo(float value){
        current.widthTo(value);
        return this;
    }
    public Sequence heightTo(float value){
        current.heightTo(value);
        return this;
    }
    public Sequence alphaTo(float value){
        current.alphaTo(value);
        return this;
    }

    // relative changes
    public Sequence xBy(float value){
        current.xBy(value);
        return this;
    }
    public Sequence yBy(float value){
        current.yBy(value);
        return this;
    }
    public Sequence widthBy(float value){
        current.widthBy(value);
        return this;
    }
    public Sequence heightBy(float value){
        current.heightBy(value);
        return this;
    }
    public Sequence alphaBy(float value){
        current.alphaBy(value);
        return this;
    }
    // scaled changes
    public Sequence scaleWidth(float value){
        current.scaleWidth(value);
        return this;
    }
    public Sequence scaleHeight(float value){
        current.scaleHeight(value);
        return this;
    }
    public Sequence scaleAlpha(float value){
        current.scaleAlpha(value);
        return this;
    }
}