package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import net.electroland.utils.ParameterMap;

public class BigShowCue extends Cue {

    private int waitMillis;
    private List<String>cues;

    public BigShowCue(ParameterMap p) {
        super(p);
        waitMillis = p.getRequiredInt("waitMillis");
        cues = p.getRequiredList("cues");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        cp.play(cues.get(new Random().nextInt(cues.size())));
    }

    @Override
    public boolean ready(EventMetaData meta) {
        return System.currentTimeMillis() - meta.getTimeOfLastNonScreenSaverCue() > waitMillis;
    }
}