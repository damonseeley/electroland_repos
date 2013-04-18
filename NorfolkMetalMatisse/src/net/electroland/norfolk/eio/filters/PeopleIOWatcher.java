package net.electroland.norfolk.eio.filters;

import java.util.ArrayList;
import java.util.Collection;

import net.electroland.eio.IOListener;
import net.electroland.eio.ValueSet;

public class PeopleIOWatcher implements IOListener {

    private Collection<PeopleListener>listeners = new ArrayList<PeopleListener>();

    @Override
    public void dataReceived(ValueSet values) {
        for (String id : values.keySet()){
            int value = values.get(id).getValue();
            if (value != 0){
                notifyListenersEntered(id, PersonEvent.Direction.UNKNOWN, PersonEvent.Behavior.ENTER);
            }
        }
    }

    public void addListener(PeopleListener pl){
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