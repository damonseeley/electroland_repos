package net.electroland.ea;

public class ChangeBy extends Change {

    public ChangeBy(float difference){
        super(difference);
    }

    @Override
    public float applyTo(float startValue) {
        return startValue + changeValue;
    }
}