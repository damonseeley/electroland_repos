package net.electroland.ea;

public class ChangeTo extends Change {

    public ChangeTo(float changeValue){
        super(changeValue);
    }

    @Override
    public float applyTo(float startValue) {
        return changeValue;
    }
}