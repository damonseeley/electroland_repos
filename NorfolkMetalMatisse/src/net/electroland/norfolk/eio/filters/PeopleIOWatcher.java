package net.electroland.norfolk.eio.filters;

import java.util.ArrayList;
import java.util.Collection;

import net.electroland.eio.IOListener;
import net.electroland.eio.ValueSet;

public class PeopleIOWatcher implements IOListener {

    public class PersonEventCodes {
        // Using these values so the virtual channel output shows up in the IOFrameTest display
        //    when an event is triggered.
        // NOTE: We don't currently send "Center" messages, but we could if we wanted to.
        public final static int NO_EVENT = 0;
        public final static int ENTER_L = 1 * (Short.MAX_VALUE/2) / 3;
        public final static int ENTER_C = 2 * (Short.MAX_VALUE/2) / 3;
        public final static int ENTER_R = 3 * (Short.MAX_VALUE/2) / 3;
        public final static int EXIT_L = -1 * (Short.MAX_VALUE/2) / 3;
        public final static int EXIT_C = -2 * (Short.MAX_VALUE/2) / 3;
        public final static int EXIT_R = -3 * (Short.MAX_VALUE/2) / 3;
    }

    private Collection<PeopleListener>listeners = new ArrayList<PeopleListener>();

    @Override
    public void dataReceived(ValueSet values) {
        for (String id : values.keySet()){
            int value = values.get(id).getValue();
            
            switch (value) {
            
            case PersonEventCodes.ENTER_L:
                notifyListenersEntered(id, PersonEvent.Direction.LEFT, PersonEvent.Behavior.ENTER);
                break;
                
            case PersonEventCodes.ENTER_C:
                notifyListenersEntered(id, PersonEvent.Direction.CENTER, PersonEvent.Behavior.ENTER);
                break;
                
            case PersonEventCodes.ENTER_R:
                notifyListenersEntered(id, PersonEvent.Direction.RIGHT, PersonEvent.Behavior.ENTER);
                break;
                
            case PersonEventCodes.EXIT_L:
                notifyListenersEntered(id, PersonEvent.Direction.LEFT, PersonEvent.Behavior.EXIT);
                break;
                
            case PersonEventCodes.EXIT_C:
                notifyListenersEntered(id, PersonEvent.Direction.CENTER, PersonEvent.Behavior.EXIT);
                break;
                
            case PersonEventCodes.EXIT_R:
                notifyListenersEntered(id, PersonEvent.Direction.RIGHT, PersonEvent.Behavior.EXIT);
                break;
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