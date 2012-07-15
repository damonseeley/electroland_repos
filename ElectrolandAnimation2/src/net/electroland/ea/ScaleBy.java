package net.electroland.ea;

public class ScaleBy extends Change {

    public ScaleBy(float factor){
        super(factor);// sets changeValue to factor
    }

    @Override
    public float applyTo(float startValue) {
        return startValue * changeValue;
    }
}