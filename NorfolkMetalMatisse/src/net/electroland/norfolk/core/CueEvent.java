package net.electroland.norfolk.core;

public class CueEvent extends NorfolkEvent {

    protected Cue sourceCue;

    public CueEvent(Cue sourceCue){
        this.sourceCue = sourceCue;
    }
}