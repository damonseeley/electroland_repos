package net.electroland.eio.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

import net.electroland.eio.IState;


public class Model {

    Collection<ModelWatcher> watchers = Collections.synchronizedList(new ArrayList<ModelWatcher>());
    Collection<ModelListener> listeners = new ArrayList<ModelListener>();

    public final void addModelWatcher(ModelWatcher watcher, String name, Collection<IState> states)
    {
        watcher.setName(name);
        watcher.setStates(states);
        watchers.add(watcher);
    }
    public final void addModelListener(ModelListener listener)
    {
        listeners.add(listener);
    }

    public final void poll()
    {
        for (ModelWatcher watcher : watchers){
            ModelEvent evt = watcher.doPoll();
            if (evt != null)
            {
                for (ModelListener listener : listeners)
                {
                    listener.modelChanged(evt);
                }
            }
        }
    }
}