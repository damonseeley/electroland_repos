package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import net.electroland.utils.ParameterMap;

public class TrainCue extends Cue {


    private List<String> shows;

    public TrainCue(ParameterMap p) {
        super(p);
        shows = p.getRequiredList("cues");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        cp.play(shows.get(new Random().nextInt(shows.size())));
    }

    @Override
    public boolean ready(EventMetaData meta) {
        // TODO Auto-generated method stub
        return false;
    }
}