package net.electroland.edmonton.core.sequencing;

import java.util.Collection;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.edmonton.core.model.LastTrippedModelWatcher;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SimpleSequencer implements Runnable, ModelListener{

    static Logger logger = Logger.getLogger(SimpleSequencer.class);

    private Map <String, Show> shows = new Hashtable<String, Show>();
    private Map <String, Object> context;
    private Thread thread;
    private Show current;
    private int clipDelay = 0;
    public String liveShowId, quietShowId;

    public static void main(String args[])
    {
        SimpleSequencer s = new SimpleSequencer("EIA-seq-LITE.properties", null);
        s.play("show1");
    }
    
    public SimpleSequencer(String propsName, Map<String,Object> context)
    {
        // requires sound manager, animation manager, model, and ELU
        this.context = context;

        ElectrolandProperties ep = new ElectrolandProperties(propsName);

        liveShowId = ep.getRequired("global", "settings", "live");
        quietShowId = ep.getRequired("global", "settings", "quiet");

        Integer clipDelay = ep.getOptionalInt("global", "settings", "clip_delay");
        if (clipDelay != null)
        {
            this.clipDelay = clipDelay;
        }

        for (String name : ep.getObjectNames("show"))
        {
            logger.info("configuring show '" + name + "'...");
            // parse cues
            HashMap <String, Cue> showCues = new HashMap<String, Cue>();
            Map<String, ParameterMap> cues = ep.getObjects(name);
            for (String cueName : cues.keySet())
            {
                int dot = cueName.indexOf('.');
                String type = cueName.substring(0,dot);
                String id = cueName.substring(dot + 1, cueName.length());

                logger.info("cue '" + id + "' of type '" + type + "'");

                Cue nCue = null;
                if (type.equals("cue"))
                {
                    nCue = new TimingCue(cues.get(cueName));
                }else if (type.equals("soundcue"))
                {
                    nCue = new SoundCue(cues.get(cueName));
                }else if (type.equals("clipcue"))
                {
                    nCue = new ClipCue(cues.get(cueName));
                }else{
                    throw new OptionException("Unknown cue type '" + type + "'");
                }

                if (nCue != null)
                {
                    nCue.id = id;
                    showCues.put(id, nCue);
                }
            }
            // connect cues
            for (Cue cue : showCues.values())
            {
                if (cue.parentName != null){
                    Cue parent = showCues.get(cue.parentName);
                    if (parent != null){
                        cue.parent = parent;
                    }else{
                        throw new OptionException("Can't find Cue '" + cue.parentName + "' in show '" + name + "'");
                    }
                }
            }
            // parse show
            Show show = new Show(ep.getParams("show", name));
            // add cues to show
            show.cues = showCues.values();
            // reset all cues
            show.reset();
            // store
            shows.put(name, show);
        }
    }

    public void play(String showName)
    {
        current = shows.get(showName);
        if (current != null)
        {
            current.reset();
            if (thread == null)
            {
                thread = new Thread(this);
                thread.start();
            }
            
        }else{
            logger.warn("No show '" + showName + "' has been definied.");
        }
    }

    public Collection<String> getSetList()
    {
    	return shows.keySet();
    }
    
    public void stop()
    {
        thread = null;
    }

    public int getClipDelay() {
        return clipDelay;
    }

    public void setClipDelay(int clipDelay) {
        this.clipDelay = clipDelay;
    }

    @Override
    public void run() {
        logger.info("sequencer started.");

        long start = System.currentTimeMillis();

        while(thread != null)
        {
            long passed = System.currentTimeMillis() - start;

            for (Cue cue : current.cues)
            {
                int time = cue.getTime();
                if (cue instanceof ClipCue)
                {
                    time += clipDelay;
                }
                if (!cue.played && passed >= time)
                {
                    logger.debug("playing " + cue.id + " at " + passed);
                    cue.play(context);
                    cue.played = true;
                }
            }

            if (passed >= current.duration)
            {
                logger.info("show over at " + passed);
                if (current.followWith != null)
                {
                    current.reset();
                    logger.info("Starting show '" + current.followWith + "'");
                    current = shows.get(current.followWith);
                    start = System.currentTimeMillis();
                }else{
                    stop();
                }
            }

            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        logger.info("sequencer stopped.");
    }

    @Override
    public void modelChanged(ModelEvent evt) {
        if (evt.getSource() instanceof LastTrippedModelWatcher){
            context.put("tripRecords",evt.optionalPostiveDetails.get(LastTrippedModelWatcher.TRIP_TIMES));
        }
    }
}