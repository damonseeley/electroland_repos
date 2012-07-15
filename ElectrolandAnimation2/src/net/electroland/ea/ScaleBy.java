package net.electroland.ea;

public class ScaleBy extends Change {

    public ScaleBy(float changeValue){
        super(changeValue);
    }

    @Override
    public float applyTo(float startValue) {
        return startValue * changeValue;
    }
}