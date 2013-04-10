package net.electroland.norfolk.eio.filters;

import java.util.ArrayList;
import java.util.Collection;

import net.electroland.eio.IOListener;
import net.electroland.eio.ValueSet;

public class PeopleIOWatcher implements IOListener {

    Collection<PeopleListener>listeners;

    @Override
    public void dataReceived(ValueSet values) {
        // TODO Interpret data from MatchFilters to determine if a new person
        // has entered.
    }

    public void addListener(PeopleListener pl){
        if (listeners == null){
            listeners = new ArrayList<PeopleListener>();
        }
        listeners.add(pl);
    }

    public void notifyListenersEntered(String channelId,
                                        PersonEvent.Direction direction,
                                        PersonEvent.Behavior behavior){
        switch(behavior){
            case ENTER:
                for (PeopleListener pl : listeners){
                    pl.personEntered(new PersonEvent(channelId, direction, behavior));
                }
                break;
            case EXIT:
                for (PeopleListener pl : listeners){
                    pl.personExited(new PersonEvent(channelId, direction, behavior));
                }
                break;
        }
    }
}