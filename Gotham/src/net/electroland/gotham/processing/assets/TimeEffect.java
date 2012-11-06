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

    public Color getColor(int colorId) {
        return colors.get(colorId);
    }

    public float getEntropy() {
        return entropy;
    }

    private void parseColors(String paletteFileName, int x, int y, int dy){

        if (paletteFileName.startsWith("east")){
            // set 1
            colors.put(0, new Color(0, 110, 150));
            colors.put(1, new Color(255, 0, 0));
            colors.put(2, new Color(255, 127, 0));
            colors.put(3, new Color(255, 230, 40));

        }else if (paletteFileName.startsWith("west")){
            // set 1
            colors.put(0, new Color(80, 0, 80));
            colors.put(1, new Color(255, 0, 50));
            colors.put(2, new Color(0, 255, 255));
            colors.put(3, new Color(0, 100, 255));
        }
    }

    public boolean isBefore(int referenceHours, int referenceMinutes)
    {
        return hours < referenceHours ||
               hours == referenceHours && referenceMinutes > minutes;
    }

    public int minutesSince(int referenceHours, int referenceMinutes){
        // it happened today e.g., current time is 12:00, effect was at 9:30
        if (isBefore(referenceHours, referenceMinutes)){
            return ((referenceHours - hours) * 60) + referenceMinutes - minutes;
        }else{
            // it happened yesterday
            // e.g., it is 0:51 and it happend at 20:00
            return ((24 - hours + referenceHours) * 60) + (referenceMinutes - minutes);
        }
    }

    public int minutesUntil(int referenceHours, int referenceMinutes){
        // 1440 = 24 hours * 60 minutes
        return 1440 - minutesSince(referenceHours, referenceMinutes);
    }

    @Override
    public int compareTo(TimeEffect in) {
        return this.isBefore(in.hours, in.minutes) ? -1 : 1;
    }

    public String toString(){
        StringBuffer sb = new StringBuffer("TimeEffect[");
        sb.append("hours=").append(hours).append(", ");
        sb.append("minutes=").append(minutes).append(", ");
        sb.append("entropy=").append(entropy).append(", ");
        sb.append("hueVariation=").append(hueVariation).append(", ");
        sb.append("colors=").append(colors).append(", ");

        return sb.append(']').toString();
    }
}