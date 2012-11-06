package net.electroland.gotham.processing.assets;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.utils.ElectrolandProperties;

public class TimeEffectSet {

    private EasingFunction easingFunction;
    private List<TimeEffect>effects;

    // unit tests
    public static void main(String args[]){

        TimeEffectSet set = new TimeEffectSet(new Linear());

        ElectrolandProperties ep = new ElectrolandProperties("Gotham-global.properties");

        for (String name : ep.getObjectNames("east")){
            if (name.startsWith("timeEffect")){
                set.add(new TimeEffect(ep.getParams("east", name)));
            }
        }

        for (TimeEffect effect : set.effects){
            System.out.println(effect);
            TimeEffect first = set.effects.get(0);
            System.out.println("effect is before first " + effect.isBefore(first.hours, first.minutes));
            System.out.println("first is effect first " + first.isBefore(effect.hours, effect.minutes));
            System.out.println("");
        }

        for (int hour = 0; hour < 24; hour++){
            int minutes = (int)(Math.random() * 60);
            Bracket b = set.getEffectBracket(hour, minutes);
            System.out.println("for " + hour + ":" + minutes  + "...");
            System.out.println("last:  " + b.prior);
            System.out.println("next: " + b.next);
            System.out.println("blended: " + set.getEffect(hour, minutes));
            System.out.println();
        }

        int[] now = TimeEffectSet.getTime(new Date());
        System.out.println(set.getEffectBracket(now[0], now[1]).prior);
        System.out.println(set.getEffectBracket(now[0], now[1]).next);

        // get color
    }

    public TimeEffectSet(EasingFunction easingFunction){
        this.easingFunction = easingFunction;
        this.effects = new ArrayList<TimeEffect>();
    }

    public void add(TimeEffect te){
        this.effects.add(te);
        Collections.sort(this.effects);
    }

    public Bracket getEffectBracket(int hours, int minutes){

        Bracket bracket = new Bracket();

        for (TimeEffect effect : effects){
            if (effect.isBefore(hours, minutes)){
                bracket.prior = effect;
            }else{
                bracket.next = effect;
                break;
            }
        }

        TimeEffect first = effects.get(0);

        if (bracket.prior == null){ // prior effect was yesterday
            bracket.prior = effects.get(effects.size() - 1);
        }
        if (bracket.next == null){ // next effect is tomorrow
            bracket.next = first;
        }

        return bracket;
    }

    /** 
     * convert date to hours, minutes (0:0 to 23:59)
     * @param date
     * @return
     */
    public static int[] getTime(Date date){
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        int hours   = calendar.get(Calendar.HOUR) + 12;
        int minutes = calendar.get(Calendar.MINUTE);
        if (hours <= 12 && calendar.get(Calendar.AM_PM) == 1){
            hours += 12;
        }
        return new int[]{hours, minutes};
    }

    public TimeEffect getEffect(Date date){
        int[] time = getTime(date);
        return getEffect(time[0], time[1]);
    }

    public TimeEffect getEffect(int hours, int minutes){

        TimeEffect newEffect = new TimeEffect(hours, minutes);
        Bracket effects = getEffectBracket(hours, minutes);

        float minPrev = effects.prior.minutesSince(hours, minutes);
        float minNext  = effects.next.minutesUntil(hours, minutes);

        float percentComplete = minPrev / (minPrev + minNext);

        float v1 = effects.prior.hueVariation;
        float v2 = effects.next.hueVariation;
        newEffect.hueVariation = easingFunction.valueAt(percentComplete, v1, v2);

        // set colors
        for (Integer cid : effects.prior.colors.keySet()){
            Color newColor = getColor(effects.prior, effects.next, 
                                      newEffect.hueVariation, 
                                      cid, percentComplete);

            newEffect.colors.put(cid, newColor);
        }

        // entropy 
        newEffect.entropy = easingFunction.valueAt(percentComplete, effects.prior.entropy, effects.next.entropy);

        return newEffect;
    }

    private Color getColor(TimeEffect prior, TimeEffect next, float variation, int cid, float percentComplete){

        Color priorColor = prior.getColor(cid);
        Color nextColor  = next.getColor(cid);

        System.out.println("blending " + cid + " at " + percentComplete + "% with variation " + variation);

        float r1 = priorColor.getRed();
        float r2 = nextColor.getRed();

        float g1 = priorColor.getGreen();
        float g2 = nextColor.getGreen();

        float b1 = priorColor.getBlue();
        float b2 = nextColor.getBlue();


        float r3 = easingFunction.valueAt(percentComplete, r1, r2);
        float g3 = easingFunction.valueAt(percentComplete, g1, g2);
        float b3 = easingFunction.valueAt(percentComplete, b1, b2);

        return new Color((int)r3, (int)g3, (int)b3);

//      doh.  hue variation really needs to be precalculated and stored persistently with each ball.
//        System.out.println("got: " + r3 + ", " + g3 + ", " + b3);
//        
//        float[] hsb = java.awt.Color.RGBtoHSB((int)r3, (int)g3, (int)b3, new float[3]);
//
//        hsb[0] = adjustHue(hsb[0], variation);
//
//        return new Color(Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]));
    }

    private float adjustHue(float hue, float variation){
        float range = hue * variation;
        return (hue - (range / 2)) + (float)(Math.random() * range);
    }

    class Bracket{
        public TimeEffect prior;
        public TimeEffect next;
    }
}