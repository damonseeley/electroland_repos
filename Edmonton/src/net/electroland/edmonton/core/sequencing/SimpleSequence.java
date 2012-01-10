package net.electroland.edmonton.core.sequencing;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;

public class SimpleSequence implements Runnable{

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
            // parse cues
            HashMap <String, Cue> pieceCues = new HashMap<String, Cue>();
            Map<String, ParameterMap> cues = ep.getObjects(name);
            for (String cueName : cues.keySet())
            {
                int dot = cueName.indexOf('.');
                String type = cueName.substring(0,dot);
                String id = cueName.substring(dot + 1, cueName.length());
                if (type.equals("cue"))
                {
                    pieceCues.put(id, new TimingCue(cues.get(cueName)));
                }else if (type.equals("soundcue"))
                {
                    pieceCues.put(id, new SoundCue(cues.get(cueName)));
                }else if (type.equals("clipcue"))
                {
                    pieceCues.put(id, new ClipCue(cues.get(cueName)));
                }
            }
            // connect cues
            for (Cue cue : pieceCues.values())
            {
                if (cue.parentName != null){
                    cue.parent = pieceCues.get(cue.parentName);
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

            if (passed > current.duration)
            {
                if (current.followWith != null)
                {
                    current.reset();
                    current = pieces.get(current.followWith);
                    start = System.currentTimeMillis();
                }else{
                    stop();
                }
            }

            for (Cue cue : current.cues)
            {
                if (!cue.played && passed > cue.getTime())
                {
                    cue.play(context);
                    cue.played = true;
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