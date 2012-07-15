package net.electroland.ea;

public class ChangeBy extends Change {

    public ChangeBy(float offset){
        super(offset);// sets changeValue to offset
    }

    @Override
    public float applyTo(float startValue) {
        return startValue + changeValue;
    }
}