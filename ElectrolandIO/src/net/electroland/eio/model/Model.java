package net.electroland.eio.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Vector;

import net.electroland.eio.IState;


public class Model {

    Collection<ModelWatcher> watchers = Collections.synchronizedList(new ArrayList<ModelWatcher>());
    Collection<ModelListener> listeners = new HashSet<ModelListener>();

    public final void addModelWatcher(ModelWatcher watcher, String name, Collection<IState> states)
    {
        watcher.setName(name);
        watcher.setStates(states);
        watchers.add(watcher);
    }
    public final void addModelWatcher(ModelWatcher watcher, String name, IState state)
    {
        watcher.setName(name);
        Vector<IState> a =  new Vector<IState>();
        a.add(state);
        watcher.setStates(a);
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