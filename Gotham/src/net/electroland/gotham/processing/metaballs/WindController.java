package net.electroland.gotham.processing.metaballs;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class WindController {

    private Wind current;
    private List<Wind>winds;
    private Random generator;
    private float minSecondsBetweenGusts, maxSecondsBetweenGusts;
    private long nextGust;

    public WindController(float minSecondsBetweenGusts, float maxSecondsBetweenGusts){
        this.minSecondsBetweenGusts = minSecondsBetweenGusts;
        this.maxSecondsBetweenGusts = maxSecondsBetweenGusts;
        winds = new ArrayList<Wind>();
        generator = new Random();
    }

    public void addWind(Wind wind){
        winds.add(wind);
    }

    public Wind next() {
        if (current == null) {
            if (System.currentTimeMillis() > nextGust){
                current = pickNextWind();
            }
        }else{
            if (current.isExpired()){
                current = null;
                nextGust = (int)(generator.nextInt((int)(maxSecondsBetweenGusts - minSecondsBetweenGusts) * 1000) + (minSecondsBetweenGusts * 1000)) + System.currentTimeMillis();
            }
        }
        return current;
    }

    public Wind pickNextWind(){
        if (winds.size() == 0){
            return null;
        }
        int nextIndex = generator.nextInt(winds.size());
        Wind next = winds.get(nextIndex);
        return next.reset();
    }
}