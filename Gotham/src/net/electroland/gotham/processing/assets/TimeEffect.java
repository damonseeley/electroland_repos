package net.electroland.gotham.processing.assets;

import java.awt.Color;
import java.util.HashMap;
import java.util.Map;

import net.electroland.utils.ParameterMap;

public class TimeEffect implements Comparable<TimeEffect> {

    protected int hours, minutes;
    protected Map<Integer,Color> colors = new HashMap<Integer, Color>();
    protected float hueVariation;
    protected float entropy;

    public TimeEffect(ParameterMap vals){

        hours = vals.getRequiredInt("hours");
        minutes = vals.getRequiredInt("minutes");
        hueVariation = vals.getRequiredDouble("hueVariation").floatValue();
        entropy = vals.getRequiredDouble("entropy").floatValue();

        int x = vals.getRequiredInt("x");
        int y = vals.getRequiredInt("y");
        int dy = vals.getRequiredInt("dy");
        String paletteName = vals.getRequired("colorPalette");
        parseColors(paletteName, x, y, dy);
    }

    public TimeEffect(int hours, int minutes){
        this.hours = hours;
        this.minutes = minutes;
    }

    private void parseColors(String paletteFileName, int x, int y, int dy){
        /**
east.color.0 = $r 0 $g 110 $b 150
east.color.1 = $r 255 $g 0 $b 0
east.color.2 = $r 255 $g 127 $b 0
east.color.3 = $r 255 $g 230 $b 40

west.color.0 = $r 80 $g 0 $b 80
west.color.1 = $r 255 $g 0 $b 50
#west.color.2 = $r 127 $g 0 $b 255
west.color.2 = $r 0 $g 255 $b 255
west.color.3 = $r 0 $g 100 $b 255
         */
        
        // TODO
    }

    public boolean isBefore(int referenceHours, int referenceMinutes)
    {
        return hours < referenceHours ||
               hours == referenceHours && minutes < referenceMinutes;
    }

    public int minutesBetween(int referenceHours, int referenceMinutes)
    {
        return (60 * (referenceHours - hours)) + referenceMinutes - minutes;
    }

    public Color getColor(int colorId) {
        return colors.get(colorId);
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

    @Override
    public int compareTo(TimeEffect in) {
        return this.minutesBetween(in.hours, in.minutes);
    }
}