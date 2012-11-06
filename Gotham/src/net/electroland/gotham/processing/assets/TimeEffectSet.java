package net.electroland.gotham.processing.assets;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import net.electroland.ea.EasingFunction;

public class TimeEffectSet {

    private EasingFunction easingFunction;
    private List<TimeEffect>effects;

    public TimeEffectSet(EasingFunction easingFunction){
        this.easingFunction = easingFunction;
        this.effects = new ArrayList<TimeEffect>();
    }

    public void add(TimeEffect te){

        effects.add(te);
        Collections.sort(effects);
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

        if (bracket.prior == null){ // e.g., you only had one effect, and it happened earlier.
            bracket.prior = effects.get(0);
        }
        if (bracket.next == null){ // e.g., wraparound at end of day
            bracket.next = effects.get(0);
        }

        return bracket;
    }

    public TimeEffect getEffect(Date date){

        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        int hours   = calendar.get(Calendar.HOUR);
        int minutes = calendar.get(Calendar.MINUTE);

        TimeEffect newEffect = new TimeEffect(hours, minutes);
        Bracket effects = getEffectBracket(hours, minutes);

        float dBefore = Math.abs(effects.prior.minutesBetween(hours, minutes));
        float dAfter  = Math.abs(effects.prior.minutesBetween(hours, minutes));

        float percentComplete = dBefore / dBefore + dAfter;

        float v1 = effects.prior.hueVariation;
        float v2 = effects.next.hueVariation;
        newEffect.setHueVariation(easingFunction.valueAt(v1, v2, percentComplete));

        // set colors
        for (Integer cid : effects.prior.colors.keySet()){
            Color newColor = getColor(effects.prior, effects.next, 
                                      newEffect.getHueVariation(), 
                                      cid, percentComplete);

            newEffect.colors.put(cid, newColor);
        }

        // entropy 
        newEffect.setEntropy(easingFunction.valueAt(effects.prior.entropy, effects.next.entropy, percentComplete));

        return newEffect;
    }

    private Color getColor(TimeEffect prior, TimeEffect next, float variation, int cid, float percentComplete){

        Color priorColor = prior.getColor(cid);
        Color nextColor  = next.getColor(cid);

        float r1 = priorColor.getRed();
        float r2 = nextColor.getRed();

        float g1 = priorColor.getGreen();
        float g2 = nextColor.getGreen();

        float b1 = priorColor.getBlue();
        float b2 = nextColor.getBlue();

        Color color = new Color(
                easingFunction.valueAt(r1, r2, percentComplete),
                easingFunction.valueAt(g1, g2, percentComplete),
                easingFunction.valueAt(b1, b2, percentComplete)
                );

        float[] hsb = java.awt.Color.RGBtoHSB((int)color.getRed(), 
                                              (int)color.getGreen(), 
                                              (int)color.getBlue(), 
                                              new float[3]);


        hsb[0] = adjustHue(hsb[0], variation);

        return new Color(Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]));
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