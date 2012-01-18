package net.electroland.ea.changes;

import net.electroland.ea.Change;
import net.electroland.ea.State;

/**
 * Pause, then jump to a new state.
 * @author production
 *
 */
public class DelayedInstantChange extends Change {

    @Override
    public State nextState(State init, double percentComplete) {
        return init;
    }
}