package net.electroland.edmonton.core.sequencing;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Map;

import net.electroland.edmonton.core.EIAClipPlayer;
import net.electroland.edmonton.core.model.Track;
import net.electroland.edmonton.core.model.TrackerModelWatcher;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class ClipCue extends Cue{
    
    final static int PER_SENSOR = 0;
    final static int PER_TRACK = 1;
    final static int GLOBAL = 2;
    
    private double x;
    private int mode = -1;
    private String clipName;
    private EIAClipPlayer clipPlayer;

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
    		
    	if (context != null)
    	{
    	    Object cpo = context.get("clipPlayer");
			if (cpo instanceof EIAClipPlayer)
			{
				EIAClipPlayer cp = (EIAClipPlayer)cpo;

				switch(mode){
            	case(GLOBAL):
            		playClipAt(cp, x);
            		break;
            	case(PER_SENSOR):
            		// record last time all IStates were tripped and start based on time since tripping
            		break;
            	case(PER_TRACK):
            		Collection<Track> c = ((TrackerModelWatcher)context.get("tracker")).getAllTracks();
            		for (Track track : c)
            		{
                		playClipAt(cp, x + track.x);
            		}
            		break;
            	}

			}else{
				System.out.println("no clipPlayer found.");
			}
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