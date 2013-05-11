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

        return cues;
    }
}