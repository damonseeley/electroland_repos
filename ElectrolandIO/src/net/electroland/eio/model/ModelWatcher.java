package net.electroland.eio.model;

import java.util.Collection;
import java.util.Map;

import net.electroland.eio.IState;

abstract public class ModelWatcher {

    private String name;
    private Collection <IState> states;

    // instantiate and populate this in your poll() method anytime you get 
    // resolved to 'true', and the instance will be passed along with your 
    // ModelEvent.
    public Map<String, Object>optionalDetails;

    // return true if the state you are watching for has occurred.  false
    // otherwise.  If true, then an event will be returned on behalf of this
    // watcher, simply containing the name of this watcher and the fact that
    // it returned true.
    abstract public boolean poll();

    public final String getName() {
        return name;
    }

    protected final void setName(String name) {
        this.name = name;
    }

    public final Collection<IState> getStates() {
        return states;
    }

    protected final void setStates(Collection<IState> states) {
        this.states = states;
    }

    protected final ModelEvent doPoll(){
        if (poll()){
            ModelEvent evt = new ModelEvent(this);
            evt.watcherName = name;
            evt.optionalPostiveDetails = optionalDetails;
            return evt;
        }else{
            return null;
        }
    }
}