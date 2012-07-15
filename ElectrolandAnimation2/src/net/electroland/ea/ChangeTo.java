package net.electroland.ea;

public class ChangeTo extends Change {

    public ChangeTo(float targetValue){
        super(targetValue);// sets changeValue to targetValue
    }

    @Override
    public float applyTo(float startValue) {
        return changeValue;
    }
}