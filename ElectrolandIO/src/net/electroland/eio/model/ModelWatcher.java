package net.electroland.eio.model;

import java.util.Collection;

import net.electroland.eio.IState;

abstract class ModelWatcher {

    String name;
    Collection <IState> states;

    public ModelWatcher(String name, Collection <IState>states)
    {
        this.name = name;
        this.states = states;
    }

    // return an event if there is one, null otherwise. (yuck: don't like this)
    // somehow this has to guarantee name is populated too.  extra yuck.  wrong
    // pattern.
    abstract public ModelEvent poll();

    public ModelEvent createEvent(){
        ModelEvent evt = new ModelEvent(this);
        evt.watcherName = name;
        return evt;
    }
}