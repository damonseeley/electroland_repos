package net.electroland.gotham.processing.metaballs;

import java.util.HashMap;
import java.util.Map;

public class MultiplierMap {

    private Map<String, Float> multipliers;

    public void addMultiplier(int ballGroupId, String value, float multiplier){
        if (multipliers == null){
            multipliers = new HashMap<String, Float>();
        }
        multipliers.put(key(ballGroupId, value), multiplier);
    }

    public float getMultiplier(int ballGroupId, String value){
        if (multipliers == null){
            return 1f;
        } else {
            Float f = multipliers.get(key(ballGroupId, value));
            return f == null ? 1f : f;
        }
    }

    public static String key(int ballGroupId, String value){
        return ballGroupId + value;
    }
}