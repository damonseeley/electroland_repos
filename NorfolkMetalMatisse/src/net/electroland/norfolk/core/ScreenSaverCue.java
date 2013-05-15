package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import net.electroland.utils.ParameterMap;

public class ScreenSaverCue extends Cue {

    boolean isSaving = false, firsttime = true;
    int timeout, fadeout;
    private List<String>shows;

    public ScreenSaverCue(ParameterMap p) {
        super(p);
        timeout = p.getRequiredInt("timeout");
        fadeout = p.getRequiredInt("fadeout");
        shows   = p.getRequiredList("cues");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        if (isSaving){
            if (firsttime){
                firsttime = false;
                cp.play(shows.get(new Random().nextInt(shows.size())));
            }
            cp.enterScreensaverMode(fadeout);
        }else{
            cp.exitScreensaverMode(fadeout);
        }
    }

    @Override
    public boolean ready(EventMetaData meta) {

        boolean everythingInactive = System.currentTimeMillis() - meta.getTimeOfLastNonScreenSaverCue() > timeout;

        if (everythingInactive && !isSaving){
            isSaving = true;
            return true;
        }else if (!everythingInactive && isSaving){
            isSaving = false;
            return true;
        }else{
            return false;
        }
    }
}