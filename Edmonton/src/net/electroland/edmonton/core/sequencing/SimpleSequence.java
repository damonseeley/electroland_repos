package net.electroland.edmonton.core.sequencing;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SimpleSequence implements Runnable{

    static Logger logger = Logger.getLogger(SimpleSequence.class);

    private Map <String, Piece> pieces = new Hashtable<String, Piece>();
    private Map <String, Object> context;
    private Thread thread;
    private Piece current;

    public static void main(String args[])
    {
        SimpleSequence s = new SimpleSequence("EIA-seq-LITE.properties", null);
        s.play("myPiece");
    }
    
    public SimpleSequence(String propsName, Map<String,Object> context)
    {
        // requires sound manager, animation manager, model, and ELU
        this.context = context;

        ElectrolandProperties ep = new ElectrolandProperties(propsName);

        for (String name : ep.getObjectNames("piece"))
        {
            logger.info("configuring piece '" + name + "'...");
            // parse cues
            HashMap <String, Cue> pieceCues = new HashMap<String, Cue>();
            Map<String, ParameterMap> cues = ep.getObjects(name);
            for (String cueName : cues.keySet())
            {
                int dot = cueName.indexOf('.');
                String type = cueName.substring(0,dot);
                String id = cueName.substring(dot + 1, cueName.length());
                logger.info("cue '" + id + "' of type '" + type + "'");
                if (type.equals("cue"))
                {
                    Cue nCue = new TimingCue(cues.get(cueName));
                    nCue.id = id;
                    pieceCues.put(id, nCue);
                }else if (type.equals("soundcue"))
                {
                    Cue nCue = new SoundCue(cues.get(cueName));
                    nCue.id = id;
                    pieceCues.put(id, nCue);
                }else if (type.equals("clipcue"))
                {
                    Cue nCue = new ClipCue(cues.get(cueName));
                    nCue.id = id;
                    pieceCues.put(id, nCue);
                }
            }
            // connect cues
            for (Cue cue : pieceCues.values())
            {
                if (cue.parentName != null){
                    Cue parent = pieceCues.get(cue.parentName);
                    if (parent != null){
                        cue.parent = parent;
                    }else{
                        throw new OptionException("Can't find Cue '" + cue.parentName + "' in piece '" + name + "'");
                    }
                }
            }
            // parse piece
            Piece piece = new Piece(ep.getParams("piece", name));
            // add cues to piece
            piece.cues = pieceCues.values();
            piece.reset();
            // store
            pieces.put(name, piece);
        }
    }

    public void play(String pieceName)
    {
        current = pieces.get(pieceName);
        if (thread == null)
        {
            thread = new Thread(this);
            thread.start();
        }
    }

    public void stop()
    {
        thread = null;
    }

    @Override
    public void run() {
        long start = System.currentTimeMillis();
        while(thread != null)
        {
            long passed = System.currentTimeMillis() - start;
            //logger.debug("passed=" + passed);

            for (Cue cue : current.cues)
            {
                if (!cue.played && passed >= cue.getTime())
                {
                    logger.debug("playing " + cue.id + " at " + passed);
                    cue.play(context);
                    cue.played = true;
                }
            }

            if (passed >= current.duration)
            {
                logger.debug("piece over.");
                if (current.followWith != null)
                {
                    current.reset();
                    logger.debug("Starting piece '" + current.followWith + "'");
                    current = pieces.get(current.followWith);
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
    }
}