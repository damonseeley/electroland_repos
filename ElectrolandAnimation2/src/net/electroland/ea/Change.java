package net.electroland.ea;

public abstract class Change {

    protected float changeValue;

    public Change(float changeValue){
        this.changeValue    = changeValue;
    }

    abstract public float applyTo(float startValue);
}