package net.electroland.edmonton.core.sequencing;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.Map;

import net.electroland.edmonton.core.EIAClipPlayer;
import net.electroland.edmonton.core.model.Track;
import net.electroland.edmonton.core.model.TrackerBasicModelWatcher;
import net.electroland.eio.IState;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class ClipCue extends Cue {

    static Logger logger = Logger.getLogger(ClipCue.class);

    final static int tolerance = 1500;
    final static int PER_SENSOR = 0;
    final static int PER_TRACK = 1;
    final static int GLOBAL = 2;

    private double x;
    private int mode = -1;
    private String clipName;

    public ClipCue(ParameterMap params)
    {
        super(params);
        String modeStr = params.getRequired("mode");
        if (modeStr.equalsIgnoreCase("PER_SENSOR")){
            mode = ClipCue.PER_SENSOR;
        }else if (modeStr.equalsIgnoreCase("PER_TRACK"))
        {
            mode = ClipCue.PER_TRACK;
        }else if (modeStr.equalsIgnoreCase("GLOBAL"))
        {
            mode = ClipCue.GLOBAL;
        }else {
            throw new OptionException("unknown mode '" + modeStr + "'");
        }

        clipName = params.getRequired("clip");
        x = params.getRequiredDouble("x");
    }

    @Override
    public void play(Map<String,Object> context) {

        if (context != null && context.get("clipPlayer") instanceof EIAClipPlayer)
        {
            // might be nice to store this.
            EIAClipPlayer cp = (EIAClipPlayer)context.get("clipPlayer");

            switch(mode){
            case(GLOBAL):
                playClipAt(cp, x);
                break;
            case(PER_SENSOR):
                if (context != null && context.get("tripRecords") != null)
                {
                    Map<IState, Long> tripRecords = (Map<IState, Long>)(context.get("tripRecords"));
                    for (IState state : tripRecords.keySet())
                    {
                        if (System.currentTimeMillis() - tripRecords.get(state) < tolerance)
                        {
                            playClipAt(cp, state.getLocation().x + x);
                        }
                    }
                }
                // record last time all IStates were tripped and start based on time since tripping
                break;
            case(PER_TRACK):
                
                if (context != null && context.get("tracker") instanceof TrackerBasicModelWatcher)
                {
                    Collection<Track> c = ((TrackerBasicModelWatcher)
                            context.get("tracker")).getAllTracks();
                    synchronized (c){
                        try{
                            for (Track track : c)
                            {
                                playClipAt(cp, x + track.x);
                            }
                        }catch(ConcurrentModificationException e){
                            logger.error(e);
                        }
                    }
                }else{
                    logger.warn("Context is missing 'tracker'");
                }

                break;
            }

        }else{
            logger.warn("Context is missing 'clipPlayer'");
        }
    }

    private void playClipAt(EIAClipPlayer cp, double x)
    {
        Method[] allMethods = cp.getClass().getDeclaredMethods();
        for (Method m : allMethods) {
            if (m.getName().equals(clipName))
            {
                try {
                    m.invoke(cp, x);
                } catch (IllegalArgumentException e) {
                    e.printStackTrace();
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}