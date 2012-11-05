package net.electroland.gotham.processing.assets;

import java.util.Date;
import java.util.HashMap;

import net.electroland.utils.ParameterMap;

public class TimeEffect {

    public int hours, minutes;
    public HashMap<Integer,Color> color;
    public float hueVariation;
    public float entropy;

    public TimeEffect(String name, ParameterMap vals){
        // TODO: implement
    }

    public int minutesFrom(Date date){
        return 0;
    }

    public HashMap<Integer, Color> getColor() {
        return color;
    }

    public void setColor(HashMap<Integer, Color> color) {
        this.color = color;
    }

    public float getHueVariation() {
        return hueVariation;
    }

    public void setHueVariation(float hueVariation) {
        this.hueVariation = hueVariation;
    }

    public float getEntropy() {
        return entropy;
    }

    public void setEntropy(float entropy) {
        this.entropy = entropy;
    }
}