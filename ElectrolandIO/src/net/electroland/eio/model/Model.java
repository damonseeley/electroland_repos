package net.electroland.eio.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;


public abstract class Model {

    Collection<ModelWatcher> watchers = Collections.synchronizedList(new ArrayList<ModelWatcher>());
    Collection<ModelListener> listeners = new ArrayList<ModelListener>();

    public void addModelWatcher(ModelWatcher watcher)
    {
        watchers.add(watcher);
    }
    public void addModelListener(ModelListener listener)
    {
        listeners.add(listener);
    }

    public void poll()
    {
        for (ModelWatcher watcher : watchers){
            ModelEvent evt = watcher.poll();
            if (evt != null)
            {
                for (ModelListener listener : listeners)
                {
                    listener.eventNoted(evt);
                }
            }
        }
    }
}