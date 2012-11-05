package net.electroland.gotham.processing.assets;

import java.util.Date;

import net.electroland.ea.EasingFunction;

public class TimeEffectSet {

    private EasingFunction ef;

    public TimeEffectSet(EasingFunction ef){
        this.ef = ef;
    }

    public void add(TimeEffect te){
        // TODO: implement
    }

    public TimeEffect getClosestBefore(Date date){
        // TODO: implement
        return null;
    }

    public TimeEffect getClosestAfter(Date date){
        // TODO: implement
        return null;
    }

    public Color getColor(Date date, int id){

        TimeEffect before = getClosestBefore(date);
        TimeEffect after = getClosestAfter(date);

        // TODO: calculate the ABSOLUTE VALUE of the number of minutes between now and before
        float dBefore = 0;
        // TODO: calculate the number of minutes between now and after
        float dAfter = 1;

        float percentComplete = dBefore / dBefore + dAfter;

        float r1 = before.getColor().get(id).r;
        float r2 = after.getColor().get(id).r;

        float g1 = before.getColor().get(id).g;
        float g2 = after.getColor().get(id).g;

        float b1 = before.getColor().get(id).b;
        float b2 = after.getColor().get(id).b;

        Color color = new Color(ef.valueAt(r1, r2, percentComplete),
                                ef.valueAt(g1, g2, percentComplete),
                                ef.valueAt(b1, b2, percentComplete));

        float variation = getHueVariation(date);

        // TODO: apply variation (remember to boundary check after adding/subtracting
        // the variance)

        return color;
    }

    public float getHueVariation(Date date){
        return 0;
    }

    public float getEntropy(Date date){
        return 0;
    }
}