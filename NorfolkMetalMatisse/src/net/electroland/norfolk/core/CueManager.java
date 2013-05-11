package net.electroland.norfolk.core;

import java.util.ArrayList;
import java.util.Collection;

import net.electroland.utils.ElectrolandProperties;

public class CueManager {

    public Collection<Cue> load(ElectrolandProperties props){
        ArrayList<Cue> cues = new ArrayList<Cue>();

        // singlets
        cues.add(new SingletCue(props.getParams("cues", "singlet")));

        // triplets
        cues.add(new TripletCue(props.getParams("cues", "triplet")));

        // trains
        cues.add(new TrainCue(props.getParams("cues", "trains")));

        // bigshows
        cues.add(new BigShowCue(props.getParams("cues", "bigshow")));

        // screensavers
        cues.add(new ScreenSaverCue(props.getParams("cues", "screensaver")));

        // timedshows
        cues.add(new TimedCue(props.getParams("cues", "timed")));

        return cues;
    }
}